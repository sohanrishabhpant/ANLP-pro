from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb
from datasets import Dataset
from tqdm import tqdm
import sacrebleu

# File paths and configurations
train_file = '../Dataset/en-hien/train.txt'
input_file = '../Dataset/en-hien/test.txt'
output_file = '../Dataset/en-hien/ft_dev_mistral_m1.txt'
lang = "English"
new_model = f'./models/mistral_fine_tuned_{lang}'

if not os.path.exists(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')

# Prepare dataset
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = f.readlines()

train_data = [line.strip() for line in train_data]
entries = []
for line in train_data:
    if '\t' not in line:
        continue
    og, hien = line.split('\t')
    entries.append(f"<s>[INST] Give the {lang} to Hinglish Translation for the following sentence:\n{lang} : {og}\nHinglish : [/INST] {hien} [END]</s>")

data_dict = {"text": entries}
dataset = Dataset.from_dict(data_dict)

# Setup model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bnb_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    f'./models/mistral_fine_tuned_{lang}',
    device_map=device,
    quantization_config=bnb_config
)

# Initialize WandB
# wandb.login()
# wandb.init(project='Mistral-7B-QLoRA-ANLP-Project', job_type="training", anonymous="allow")

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(f'./models/mistral_fine_tuned_{lang}')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# Prepare for training with LoRA
model.config.use_cache = False
model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
# )
# model = get_peft_model(model, peft_config)

# # Training arguments
# training_arguments = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=2,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=2,
#     save_steps=100,
#     logging_steps=25,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     bf16=False,
#     max_grad_norm=0.3,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant",
#     report_to="wandb"
# )

# from trl import SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     max_seq_length=100,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing=False,
# )

# trainer.train()
# trainer.model.save_pretrained(new_model)
# tokenizer.save_pretrained(new_model)

# # Finalize WandB
# wandb.finish()

# Inference pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=250)

# Generate translations and calculate BLEU score
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines=lines[:100]

original_sentences = []
reference_translations = []
generated_translations = []

for line in tqdm(lines, desc="Generating translations", unit="sentence"):
    if '\t' not in line:
        continue
    og_sent, hien_sent = line.strip().split('\t')
    if len(og_sent) > 120:  # Skip overly long sentences
        continue
    
    result = pipe(f"<s>[INST] Give the {lang} to Hinglish Translation for the following sentence:\n{lang} : {og_sent}\nHinglish : [/INST]")
    generated_text = result[0]['generated_text'].replace('\n', '').strip()
    if "Hinglish :" in generated_text:
        # generated_text = generated_text.split("Hinglish :", 1)[1].strip()
        generated_text = generated_text.split("[/INST]", 1)[1].split("[END]", 1)[0].strip()
    original_sentences.append(og_sent)
    reference_translations.append(hien_sent)
    generated_translations.append(generated_text)
    print(generated_text)
    # Save to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Source:{og_sent}\nTrue Translation:{hien_sent}\n Generated Translation:{generated_text}\n\n")

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(generated_translations, [reference_translations])
print(f"BLEU Score: {bleu.score:.2f}")
