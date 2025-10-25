import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data_prepare import ReviewDataModule, ReviewDataset
from models import SentimentClassifier

def train_sentiment_model(data_path='data/reviews_processed.csv', model_name='distilbert-base-uncased', n_epochs=5, sample_size: int = None):
    """
    Main function to train the sentiment analysis model on the Amazon Reviews dataset.

    Args:
        data_path (str): Path to the processed data file.
        model_name (str): Name of the transformer model to use.
        n_epochs (int): Maximum number of epochs for training.
        sample_size (int, optional): The number of reviews to use for training.
                                     If None, the full dataset is used.
    """
    # --- 1. Hyperparameters ---
    BATCH_SIZE = 64
    MAX_TOKEN_LEN = 256
    LEARNING_RATE = 2e-5
    N_CLASSES = 2  # Negative, Positive

    # --- 2. Initialize DataModule ---
    print("Initializing ReviewDataModule...")
    review_datamodule = ReviewDataModule(
        data_path=data_path,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN,
        model_name=model_name,
        sample_size=sample_size # Pass the sample size to the datamodule
    )
    review_datamodule.setup()

    n_training_steps = len(review_datamodule.train_dataloader()) * n_epochs
    n_warmup_steps = int(n_training_steps * 0.1)

    # --- 3. Initialize Model ---
    print("Initializing SentimentClassifier model...")
    model = SentimentClassifier(
        model_name=model_name,
        n_classes=N_CLASSES,
        learning_rate=LEARNING_RATE,
        n_warmup_steps=n_warmup_steps,
        n_training_steps=n_training_steps
    )

    # --- 4. Configure Training Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="sentiment-binary-best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger("lightning_logs", name="sentiment-classifier-binary")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    # --- 5. Initialize Trainer ---
    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=n_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )

    # --- 6. Start Training ---
    print(f"Starting training with {model_name} for up to {n_epochs} epochs...")
    trainer.fit(model, review_datamodule)

    # --- 7. Evaluate on Test Set and Generate Confusion Matrix ---
    print("\nTraining complete. Evaluating on the test set...")
    trainer.test(model, datamodule=review_datamodule)

    predictions = trainer.predict(model, datamodule=review_datamodule)
    if predictions:
        all_preds = torch.cat(predictions).cpu().numpy()
        true_labels = review_datamodule.test_df.sentiment.to_numpy()
        target_names = ['Negative', 'Positive'] # Updated labels

        cm = confusion_matrix(true_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix for Sentiment Analysis')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


        
if __name__ == "__main__":
    data_path = "data/reviews_processed.csv"
    train_sentiment_model(data_path=data_path, sample_size=100000)