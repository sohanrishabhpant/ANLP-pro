import logging
import os
from pathlib import Path
import torch
import pandas as pd
import wandb
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from sacrebleu.metrics import BLEU

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class DataManager:
    """Handles loading and preprocessing of translation data."""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_data(self, split: str) -> list:
        """Load data from a specific split (train/dev/test)."""
        file_path = self.data_dir / f"{split}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
            return [line.split("\t") for line in data if line and "\t" in line]
        except Exception as e:
            logging.error(f"Error loading {split} data: {e}")
            raise

class MT5Translator:
    """Main MT5 translation model class."""
    def __init__(
        self,
        model_name: str = "google/mt5-small",
        max_length: int = 128,
        experiment_name: str = "mt5_translation"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.experiment_name = experiment_name
        self.bleu = BLEU(effective_order=True)
        
        # Initialize model and tokenizer
        logging.info(f"Using device: {self.device}")
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )

    def preprocess_data(self, examples):
        """Tokenize source and target text."""
        inputs = self.tokenizer(
            examples["source"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            labels = self.tokenizer(
                examples["target"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    def evaluate_bleu(self, test_dataset):
        """Evaluate BLEU score on the test dataset."""
        logging.info("Evaluating BLEU score on the test dataset.")
        predictions, references = [], []

        for example in test_dataset:
            # Translate the source text
            prediction = self.translate(example["source"])
            predictions.append(prediction)
            references.append(example["target"])

        # Calculate BLEU score
        bleu_score = self.bleu.corpus_score(predictions, [references])
        
        logging.info(f"BLEU score on test set: {bleu_score.score:.2f}")
        return bleu_score

    def translate(self, text: str) -> str:
        """Translate a single piece of text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(
        self,
        dataset: DatasetDict,
        output_dir: str = "./results",
        num_epochs: int = 15,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        wandb_entity: str = None
    ):
        """Train the translation model."""
        try:
            # Initialize wandb
            if wandb_entity:
                wandb.init(
                    project="mt5_translation",
                    name=self.experiment_name,
                    entity=wandb_entity,
                    config={
                        "model_name": self.model.name_or_path,
                        "max_length": self.max_length,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate
                    }
                )
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                logging_steps=50,
                save_steps=200,
                eval_steps=200,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                gradient_accumulation_steps=2,
                warmup_steps=100,
                weight_decay=0.01,
                report_to="wandb" if wandb_entity else "none"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["val"],
                data_collator=self.data_collator
            )
            
            # Train model
            trainer.train()
            return trainer
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
        finally:
            if wandb.run:
                wandb.finish()

def main():
    """Main function to run the translation pipeline."""
    try:
        # Initialize data manager and load data
        data_manager = DataManager("../Dataset/en-hien")
        
        # Load and prepare datasets
        datasets = {}
        for split in ["train", "dev", "test"]:
            data = data_manager.load_data(split)
            df = pd.DataFrame(data, columns=["source", "target"])
            datasets[split.replace("dev", "val")] = Dataset.from_pandas(df)
        
        dataset_dict = DatasetDict(datasets)
        
        # Initialize and train translator
        translator = MT5Translator(experiment_name="mt5_en_hien_translation")
        
        # Preprocess datasets
        tokenized_datasets = dataset_dict.map(
            translator.preprocess_data,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        # Train model
        trainer = translator.train(
            tokenized_datasets,
            output_dir="./mt5_translation_results",
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-4,
            wandb_entity="sohan-mupparapu-iiit-hyderabad"
        )
        
        # Save model
        output_path = Path("./models/mt5_m1")
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path))
        translator.tokenizer.save_pretrained(str(output_path))
        logging.info(f"Model saved to {output_path}")
        
        # Evaluate BLEU on the test set
        bleu_score = translator.evaluate_bleu(datasets["test"])
        logging.info(f"Final BLEU score on test set: {bleu_score.score:.2f}")
        
        # Additional step: Translation and evaluation on the test set
        logging.info("Testing translations on the test set...")

        # Translate and compare the true values
        true_values = []
        translated_values = []
        
        for example in datasets["test"]:
            # Translate the source text
            translation = translator.translate(example["source"])
            true_values.append(example["target"])
            translated_values.append(translation)
        
        # Write results to a text file
        output_file = "translations_results.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for src, true, pred in zip(datasets["test"]["source"], true_values, translated_values):
                f.write(f"Source: {src}\nTrue Translation: {true}\nGenerated Translation: {pred}\n\n")

        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
