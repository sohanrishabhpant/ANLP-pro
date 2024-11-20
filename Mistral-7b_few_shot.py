import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights
import os
import sacrebleu
from tqdm import tqdm  # Import tqdm for progress bars

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
bnb_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

train_file = '../Dataset/en-hien/train.txt'
input_file = '../Dataset/en-hien/test.txt'

def few_shot_prompt(eng_sentence, num_sents=3):
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    random_lines = np.random.choice(lines, num_sents)
    
    prompt = f"Translate the following English sentence to Hinglish: \n"
    
    for line in random_lines:
        en_sent, hien_sent = line.strip().split('\t')
        prompt += f"Text: {en_sent}\nTranslation: {hien_sent}\n\n"
        
    prompt += f"Text: {eng_sentence}\nTranslation: "
    
    return prompt

def prompt_model(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=100, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    
    # Extract the part of the text after "Translation: "
    if "Translation: " in output_text:
        output_text = output_text.split("Translation: ", 1)[1].strip()
    
    return output_text

# Method 1 -> English to Hinglish, Script: Roman to Roman + Deva
output_file = '../Dataset/en-hien/prompting_dev_mistral_m1.txt'
if not os.path.exists(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')

m1_outputs = []
m1_references = []
with open(input_file, 'r') as f:
    sents = f.readlines()
    sents = [sent.strip() for sent in sents][:100]
    
    for sent in tqdm(sents, desc="Processing Method 1 (English to Hinglish)"):
        en_sent, hien_sent = sent.split('\t')
        prompt = few_shot_prompt(en_sent)
        output_text = prompt_model(prompt)
        m1_outputs.append(output_text)
        m1_references.append(hien_sent)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Source:{en_sent}\nTrue Translation:{hien_sent}\nGenerated Translation:{output_text}\n\n")

# Calculate BLEU score for Method 1
m1_bleu = sacrebleu.corpus_bleu(m1_outputs, [m1_references])
print(f"BLEU score for Method 1: {m1_bleu.score:.2f}")

# Method 2 -> English to Hindi to Hinglish, Script: Roman to Deva to Roman + Deva
train_file = '../Dataset/hi-hien/train.txt'
input_file = '../Dataset/hi-hien/test.txt'
output_file = '../Dataset/hi-hien/prompting_dev_mistral_m2.txt'

if not os.path.exists(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')

m2_outputs = []
m2_references = []
with open(input_file, 'r') as f:
    sents = f.readlines()
    sents = [sent.strip() for sent in sents][:100]
    
    for sent in tqdm(sents, desc="Processing Method 2 (English to Hindi to Hinglish)"):
        en_sent, hien_sent = sent.split('\t')
        prompt = few_shot_prompt(en_sent)
        output_text = prompt_model(prompt)
        m2_outputs.append(output_text)
        m2_references.append(hien_sent)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Source:{en_sent}\nTrue Translation:{hien_sent}\nGenerated Translation:{output_text}\n\n")

# Calculate BLEU score for Method 2
m2_bleu = sacrebleu.corpus_bleu(m2_outputs, [m2_references])
print(f"BLEU score for Method 2: {m2_bleu.score:.2f}")
