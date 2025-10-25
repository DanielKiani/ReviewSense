import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os

def explore_and_preprocess_reviews(
    train_path='data/train.csv', 
    test_path='data/test.csv',
    output_dir='data'
):
    """
    Loads the Amazon Sentiment Analysis dataset (https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
    (you need to extract the train/test splits from the zip file in the data folder),
    performs basic EDA, and preprocesses it for model training.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.
        output_dir (str): Directory to save the processed file.
    """
    # --- 1. Load Data ---
    # This dataset typically comes without headers. We'll assign them.
    # Column 1: Sentiment (1 = Negative, 2 = Positive)
    # Column 2: Title
    # Column 3: Review Text
    print(f"Loading data from '{train_path}' and '{test_path}'...")
    try:
        col_names = ['sentiment_orig', 'title', 'review']
        train_df = pd.read_csv(train_path, header=None, names=col_names)
        test_df = pd.read_csv(test_path, header=None, names=col_names)
        
        # Combine for unified EDA and preprocessing
        df = pd.concat([train_df, test_df], ignore_index=True)

    except FileNotFoundError:
        print(f"\nERROR: Make sure '{train_path}' and '{test_path}' are in the specified directory.")
        print("This script is designed for the 'Amazon Reviews for Sentiment Analysis' dataset from Kaggle.")
        return

    df.dropna(inplace=True)

    # --- 2. Preprocessing ---
    print("\n--- Preprocessing Data for Sentiment Analysis ---")

    # a) Create new sentiment labels (0 = Negative, 1 = Positive)
    # This dataset is binary, not three-class like the previous one.
    df['sentiment'] = df['sentiment_orig'].apply(lambda x: 0 if x == 1 else 1)

    # b) Combine title and review body
    df['full_text'] = df['title'].astype(str) + ". " + df['review'].astype(str)

    # c) Select and rename columns
    processed_df = df[['full_text', 'sentiment']].copy()

    # --- 4. Save Processed Data ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reviews_processed.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(processed_df)} processed reviews to '{output_path}'")

class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for Amazon Reviews.

    This class takes a pandas DataFrame of review data, a tokenizer, and a max
    token length, and prepares it for use in a PyTorch model. It handles the
    tokenization of the text and the formatting of the labels for each item.

    Attributes:
        tokenizer: The Hugging Face tokenizer to use for processing text.
        data (pd.DataFrame): The DataFrame containing the review data.
        max_token_len (int): The maximum sequence length for the tokenizer.
    """
    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len: int):
        """
        Initializes the ReviewDataset.

        Args:
            data (pd.DataFrame): The input DataFrame containing 'full_text' and
                                 'sentiment' columns.
            tokenizer: The pre-trained tokenizer instance.
            max_token_len (int): The maximum length for tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Retrieves one sample from the dataset at the specified index.

        This method handles the tokenization of a single review text, including
        padding and truncation, and formats the output into a dictionary of
        tensors ready for the model.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized inputs and the label,
                  with the following keys:
                  - 'input_ids': The token IDs of the review text.
                  - 'attention_mask': The attention mask for the review text.
                  - 'labels': The sentiment label as a tensor.
        """
        data_row = self.data.iloc[index]
        text = str(data_row.full_text)
        labels = data_row.sentiment

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels, dtype=torch.long)
        )

class ReviewDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle the Amazon Reviews dataset.

    This class encapsulates all the steps needed to process the data:
    loading, splitting, and creating PyTorch DataLoaders for training,
    validation, and testing. It allows for using a smaller random sample of the
    full dataset for faster experimentation.

    Attributes:
        data_path (str): Path to the processed CSV file.
        batch_size (int): The size of each data batch.
        max_token_len (int): The maximum sequence length for the tokenizer.
        tokenizer: The Hugging Face tokenizer instance.
        num_workers (int): The number of CPU cores to use for data loading.
        sample_size (int, optional): The number of samples to use. If None,
                                     the full dataset is used.
    """
    def __init__(self, data_path: str, batch_size: int = 16, max_token_len: int = 256, model_name='distilbert-base-uncased', num_workers: int = 0, sample_size: int = None):
        """
        Initializes the ReviewDataModule.

        Args:
            data_path (str): The path to the processed CSV data file.
            batch_size (int): The number of samples per batch.
            max_token_len (int): Maximum length of tokenized sequences.
            model_name (str): The name of the pre-trained model to use for the tokenizer.
            num_workers (int): Number of subprocesses to use for data loading.
            sample_size (int, optional): If specified, a random sample of this
                                         size will be used from the dataset.
                                         Defaults to None, which uses the full dataset.
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_workers = num_workers
        self.sample_size = sample_size
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self, stage=None):
        """
        Loads and splits the data for training, validation, and testing.

        This method is called by PyTorch Lightning. It reads the CSV, handles
        missing values, optionally takes a random sample, and performs a
        stratified train-validation-test split. The indices of the resulting
        DataFrames are reset to prevent potential KeyErrors during data loading.
        """
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)

        # If a sample size is provided, sample the dataframe
        if self.sample_size:
            print(f"Using a sample of {self.sample_size} reviews.")
            df = df.sample(n=self.sample_size, random_state=42)

        # Stratified split to maintain label distribution
        train_val_df, self.test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df.sentiment)
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df.sentiment)

        # Reset indices to prevent KeyErrors
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        print(f"Size of training set: {len(self.train_df)}")
        print(f"Size of validation set: {len(self.val_df)}")
        print(f"Size of test set: {len(self.test_df)}")

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            ReviewDataset(self.train_df, self.tokenizer, self.max_token_len),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            ReviewDataset(self.val_df, self.tokenizer, self.max_token__len),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            ReviewDataset(self.test_df, self.tokenizer, self.max_token_len),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
if __name__ == "__main__":
    
    #--- Step 1: Preprocess the Reviews Dataset ---
    print("\n--- Preprocessing started ---")
    explore_and_preprocess_reviews()
    print("\n--- Preprocessing finished ---")
    # --- Configuration ---
    data_path = "data/reviews_processed.csv"
    BATCH_SIZE = 64
    MAX_TOKEN_LEN = 256
    
    print("Initializing ReviewDataModule...")
    review_datamodule = ReviewDataModule(
        data_path=data_path,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN,
        model_name='distilbert-base-uncased',
        sample_size=100000 # Pass the sample size to the datamodule
    )
    review_datamodule.setup()

    # Fetch one batch from the training dataloader to inspect its contents
    print("\n--- Fetching one batch from the training dataloader ---")
    train_batch = next(iter(review_datamodule.train_dataloader()))
    
    print("\n--- Example Batch ---")
    print(f"Input IDs shape: {train_batch['input_ids'].shape}")
    print(f"Attention Mask shape: {train_batch['attention_mask'].shape}")
    print(f"Labels: {train_batch['labels']}")
    print(f"Labels shape: {train_batch['labels'].shape}")