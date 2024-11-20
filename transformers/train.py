import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
from decoder import build_transformer
# Hyperparameters
SRC_SEQ_LEN = 800
TGT_SEQ_LEN = 800
BATCH_SIZE = 5
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
LEARNING_RATE = 1e-5

# Helper functions and classes
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_sentence = self.src_sentences[index]
        tgt_sentence = self.tgt_sentences[index]
        src_token_ids = self.tokenize_sentence(src_sentence, self.src_vocab)
        tgt_token_ids = self.tokenize_sentence(tgt_sentence, self.tgt_vocab, target=True)
        return torch.tensor(src_token_ids), torch.tensor(tgt_token_ids)

    @staticmethod
    def tokenize_sentence(sentence, vocab, target=False):
        sentence = sentence.lower()
        tokens = [SOS_TOKEN] + word_tokenize(sentence) + [EOS_TOKEN] if target else word_tokenize(sentence)
        return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch


def build_vocab(sentences):
    word_freq = Counter()
    for sentence in sentences:
        sentence = sentence.lower()
        word_freq.update(word_tokenize(sentence))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
    idx = 4
    for word in word_freq:
        vocab[word] = idx
        idx += 1
    return vocab, len(vocab)


def create_masks(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(-1)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).type_as(tgt_mask)
    causal_mask = causal_mask == 0
    tgt_mask = tgt_mask & causal_mask
    return src_mask, tgt_mask


def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in data_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        src_mask, tgt_mask = create_masks(src_batch, tgt_input)

        optimizer.zero_grad()
        encoder_output = model.encode(src_batch, src_mask)
        output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        output = model.project(output)
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    references = []
    translations = []
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            src_mask, tgt_mask = create_masks(src_batch, tgt_input)

            encoder_output = model.encode(src_batch, src_mask)
            output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            output = model.project(output)
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

            predicted_tokens = output.argmax(dim=-1).view(tgt_batch.size(0), -1)
            translations.extend(predicted_tokens.tolist())
            references.extend(tgt_batch[:, 1:].tolist())

    # bleu_score = calculate_bleu(translations, references)
    # rouge_score = calculate_rouge(translations, references)
    # print(f"BLEU Score: {bleu_score:.4f}, ROUGE Score: {rouge_score['rouge1']:.4f}")

    return total_loss / len(data_loader)


# def calculate_bleu(predictions, references):
#     smoothing_function = SmoothingFunction().method4
#     total_bleu = 0
#     count = 0
#     for pred, ref in zip(predictions, references):
#         pred = [str(token) for token in pred]
#         ref = [str(token) for token in ref]
#         bleu = sentence_bleu([ref], pred, smoothing_function=smoothing_function)
#         total_bleu += bleu
#         count += 1
#     return total_bleu / count if count > 0 else 0.0


# def calculate_rouge(predictions, references):
#     if not predictions or not references:
#         return {"rouge1": 0.0, "rougeL": 0.0}

#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     total_rouge1 = 0
#     total_rougeL = 0
#     count = len(predictions)

#     for pred, ref in zip(predictions, references):
#         pred_str = " ".join([str(token) for token in pred])
#         ref_str = " ".join([str(token) for token in ref])
#         scores = scorer.score(ref_str, pred_str)
#         total_rouge1 += scores['rouge1'].fmeasure
#         total_rougeL += scores['rougeL'].fmeasure

#     avg_rouge1 = total_rouge1 / count
#     avg_rougeL = total_rougeL / count
#     return {"rouge1": avg_rouge1, "rougeL": avg_rougeL}


def run_experiment(model_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")

    # Load your training and validation datasets here
    with open("../../Dataset/en-hien/train.txt") as f:
        eng_train,hien_train = f.readlines().split("\t")
    with open("../../Dataset/en-hien/dev.txt") as f:
        eng_dev,hien_dev = f.readlines().split("\t")
    

    src_vocab, src_vocab_size = build_vocab(eng_train)
    tgt_vocab, tgt_vocab_size = build_vocab(hien_train)

    train_dataset = TranslationDataset(eng_train, hien_train, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    dev_dataset = TranslationDataset(eng_dev, hien_dev, src_vocab, tgt_vocab)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    model = build_transformer(src_vocab_size, tgt_vocab_size, SRC_SEQ_LEN, TGT_SEQ_LEN, model_params["D_MODEL"], model_params["N_LAYERS"], model_params["N_HEAD"], model_params["DROPOUT"], model_params["D_FF"])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss(ignore_index=src_vocab[PAD_TOKEN])

    num_epochs = 3
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, dev_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "./models/transformer.pt")

    return train_losses, val_losses


def main():
    # Experiment with different hyperparameters
    hyperparameter_combinations = [
        # {'D_MODEL': 512, 'N_LAYERS': 4, 'N_HEAD': 4, 'DROPOUT': 0.1, 'D_FF': 2048},
        {'D_MODEL': 768, 'N_LAYERS': 6, 'N_HEAD': 8, 'DROPOUT': 0.1, 'D_FF': 3072},
        # {'D_MODEL': 256, 'N_LAYERS': 3, 'N_HEAD': 2, 'DROPOUT': 0.1, 'D_FF': 1024}  # Additional hyperparameter combination
    ]

    for i, params in enumerate(hyperparameter_combinations):
        print(f"\nRunning experiment {i + 1} with params: {params}")
        train_losses, val_losses = run_experiment(params)
        print(f"train loss:{train_losses} val_loss:{val_losses}")

if __name__ == "__main__":
    main()
