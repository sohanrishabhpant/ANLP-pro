import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import pandas as pd
import logging
from pathlib import Path
import os
import sacrebleu
import json
from datetime import datetime
import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TranslationDataManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def get_data(self, type: str):
        """Load and parse data from files."""
        file_path = self.data_dir / f"{type}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
            return [
                sentences.split("\t")
                for sentences in data
                if sentences and "\t" in sentences
            ]
        except Exception as e:
            logging.error(f"Error loading {type} data: {str(e)}")
            raise

class IndicTranslator:
    def __init__(
        self, 
        model_name: str = "ai4bharat/IndicBART",
        max_length: int = 128,
        experiment_name: str = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.experiment_name = experiment_name or f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logging.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Create results directory
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        
    def preprocess_function(self, examples):
        """Tokenize inputs and targets."""
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
        
        labels = labels["input_ids"].numpy()
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    def train(
        self,
        tokenized_dataset: DatasetDict,
        output_dir: str = "./checkpoints",
        num_epochs: int = 40,
        batch_size: int = 20,
        learning_rate: float = 1e-4,
        wandb_entity: str = None
    ):
        """Train the model with WandB integration."""
        # Initialize WandB if entity is provided
        if wandb_entity:
            wandb.init(
                project="indic_translation_hi",
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

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="wandb" if wandb_entity else None
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"]
        )
        
        try:
            trainer.train()
            return trainer
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
        finally:
            if wandb_entity:
                wandb.finish()

    def translate(self, text: str):
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
                no_repeat_ngram_size=2,
            )
        
        translation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        translation = translation.strip()
        if '[SEP]' in translation:
            translation = translation.split('[SEP]')[0].strip()
        if '[CLS]' in translation:
            translation = translation.split('[CLS]')[1].strip()
        
        return translation

    def evaluate_test_set(self, test_dataset):
        """Evaluate the model on test set and save results."""
        predictions = []
        references = []
        
        # Open file to write results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"translation_results_{timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_dataset):
                source_text = example["source"]
                target_text = example["target"]
                predicted_text = self.translate(source_text)
                
                predictions.append(predicted_text)
                references.append(target_text)
                
                # Write source, prediction and target to file
                f.write(f"Source: {source_text}\n")
                f.write(f"Translation: {predicted_text}\n")
                f.write(f"Target: {target_text}\n")
                f.write("-" * 50 + "\n")
                
                if idx % 100 == 0:
                    logging.info(f"Processed {idx} test examples")

        # Calculate and return BLEU score
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        logging.info(f"BLEU Score: {bleu.score:.2f}")
        return bleu.score

def main():
    # Initialize data manager and load data
    data_manager = TranslationDataManager("../Dataset/hi-hien")
    
    try:
        # Load datasets
        train_data = data_manager.get_data("train")
        val_data = data_manager.get_data("dev")
        test_data = data_manager.get_data("test")
        
        # Create DataFrames and datasets
        df = {
            "train": pd.DataFrame(train_data, columns=["source", "target"]),
            "val": pd.DataFrame(val_data, columns=["source", "target"]),
            "test": pd.DataFrame(test_data, columns=["source", "target"])
        }
        
        ds = DatasetDict({
            name: Dataset.from_pandas(df[name]) 
            for name in ["train", "val", "test"]
        })
        
        # Initialize translator and train
        translator = IndicTranslator(experiment_name="ind_translation_experiment")
        tokenized_datasets = ds.map(
            translator.preprocess_function,
            batched=True,
            remove_columns=ds["train"].column_names
        )
        
        # Train the model with WandB tracking
        trainer = translator.train(
            tokenized_datasets,
            wandb_entity="sohan-mupparapu-iiit-hyderabad"  # Replace with your WandB entity
        )
        
        output_path = Path("./models/Indic_m2")
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path))
        
        # Evaluate on test set
        bleu_score = translator.evaluate_test_set(ds["test"])
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()