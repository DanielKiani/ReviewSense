import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoConfig
from torch.optim import AdamW
import torch
from torchmetrics.functional import accuracy
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, pipeline

class SentimentClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for the sentiment classification model.
    """
    def __init__(self, model_name='distilbert-base-uncased', n_classes=2, learning_rate=2e-5, n_warmup_steps=0, n_training_steps=0, dropout_prob=0.2): # Added dropout
        super().__init__()
        self.save_hyperparameters()

        # Configure dropout
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        config.num_labels = n_classes

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        self.log("train_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        preds = torch.argmax(output.logits, dim=1)
        val_acc = accuracy(preds, batch['labels'], task='binary')
        self.log("val_loss", output.loss, prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc, prog_bar=True, logger=True)
        return output.loss

    def test_step(self, batch, batch_idx):
        output = self.forward(**batch)
        preds = torch.argmax(output.logits, dim=1)
        test_acc = accuracy(preds, batch['labels'], task='binary')
        self.log("test_accuracy", test_acc)
        return test_acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        return torch.argmax(output.logits, dim=1)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.n_warmup_steps,
            num_training_steps=self.hparams.n_training_steps
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

class ReviewSummarizer:
    """
    A class to handle the summarization of product reviews using a pre-trained T5 model.
    """
    def __init__(self, model_name='t5-small'):
        """
        Initializes the summarizer with a pre-trained T5 model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained T5 model to use.
        """
        print(f"Loading summarization model: {model_name}...")
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the tokenizer and model from Hugging Face
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        print("Summarization model loaded successfully.")

    def summarize(self, text: str, max_length: int = 50, min_length: int = 10) -> str:
        """
        Generates a summary for a given text.

        Args:
            text (str): The review text to summarize.
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.

        Returns:
            str: The generated summary.
        """
        if not text or not isinstance(text, str):
            return ""

        # T5 models require a prefix for the task. For summarization, it's "summarize: "
        preprocess_text = f"summarize: {text.strip()}"

        # Tokenize the input text
        tokenized_text = self.tokenizer.encode(preprocess_text, return_tensors="pt").to(self.device)

        # Generate the summary
        summary_ids = self.model.generate(
            tokenized_text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode the summary and return it
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class AspectAnalyzer:
    """
    A class to handle Aspect-Based Sentiment Analysis (ABSA) using a pre-trained model.
    """
    # Changed to a different, currently valid lightweight model for ABSA.
    def __init__(self, model_name='yangheng/deberta-v3-base-absa-v1.1', force_cpu=False):
        """
        Initializes the ABSA pipeline with a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained ABSA model.
            force_cpu (bool): If True, forces the model to run on the CPU.
        """
        print(f"Loading Aspect-Based Sentiment Analysis model: {model_name}...")
        self.model_name = model_name

        if force_cpu:
            self.device = -1 # Use -1 for CPU in pipeline
            print("Forcing ABSA model to run on CPU.")
        else:
            self.device = 0 if torch.cuda.is_available() else -1

        print(f"Using device: {self.device} (0 for GPU, -1 for CPU)")

        self.absa_pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device
        )
        print("ABSA model loaded successfully.")

    def analyze(self, text: str, aspects: list) -> dict:
        """
        Analyzes the sentiment towards a list of aspects within a given text.
        """
        if not text or not isinstance(text, str) or not aspects:
            return {}

        # The model expects the review and aspect separated by a special token.
        # Note: Different ABSA models might expect different input formats.
        # This format is common but may need adjustment for other models.
        inputs = [f"{text} [SEP] {aspect}" for aspect in aspects]
        results = self.absa_pipeline(inputs)

        # Process results into a user-friendly dictionary
        aspect_sentiments = {}
        for aspect, result in zip(aspects, results):
            aspect_sentiments[aspect] = {'sentiment': result['label'], 'score': result['score']}

        return aspect_sentiments

class FineTunedSentimentClassifier:
    """
    This class handles loading the fine-tuned checkpoint and making predictions.
    """
    def __init__(self, checkpoint_path, model_name='distilbert-base-uncased', force_cpu=False):
        self.device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading fine-tuned sentiment model from checkpoint: {checkpoint_path}...")
        print(f"Using device: {self.device}")

        self.model = SentimentClassifier.load_from_checkpoint(checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labels = ['NEGATIVE', 'POSITIVE']
        print("Fine-tuned sentiment model loaded successfully.")

    def classify(self, text: str) -> dict:
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128,
            return_token_type_ids=False, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        return {'label': self.labels[prediction_idx], 'score': probabilities[0][prediction_idx].item()}

class AspectExtractor:
    """
    This class uses a Part-of-Speech (POS) tagging model to first extract all
    potential aspect terms (nouns) from a review text. It then filters these
    nouns against a pre-defined dictionary of valid aspects for a given
    product category to return only the relevant features.
    """
    def __init__(self, model_name="vblagoje/bert-english-uncased-finetuned-pos", force_cpu=False):
        self.model_name = model_name
        self.device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Part-of-Speech (POS) tagging model: {self.model_name}...")
        print(f"Using device: {self.device}")

        self.pipeline = pipeline(
            "token-classification",
            model=self.model_name,
            device=-1 if self.device == 'cpu' else 0,
            aggregation_strategy="simple"
        )
        print("POS tagging model loaded successfully.")

    def extract(self, text: str, aspect_dictionary: list) -> list:
        """
        Extracts aspects from the given text that are present in the provided
        aspect dictionary.

        Args:
            text (str): The review text to analyze.
            aspect_dictionary (list): A list of valid, known aspects for the
                                      product category.

        Returns:
            list: A list of aspects that were both found in the text and are
                  present in the aspect dictionary.
        """
        if not text or not aspect_dictionary:
            return []

        # 1. Extract all nouns from the text using the POS model
        model_outputs = self.pipeline(text)
        noun_tags = {'NOUN', 'PROPN'}
        extracted_nouns = {
            output['word'].lower() for output in model_outputs
            if output['entity_group'] in noun_tags
        }

        # 2. Filter the extracted nouns against the provided dictionary
        # We find the intersection between the two sets.
        valid_aspects = {aspect.lower() for aspect in aspect_dictionary}

        final_aspects = list(extracted_nouns.intersection(valid_aspects))

        return final_aspects
    