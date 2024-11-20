import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from decoder import build_transformer  # Ensure your decoder module is in the same directory
from nltk.tokenize import word_tokenize
# import sacrebleu
from sacrebleu.metrics import BLEU


# Hyperparameters
BATCH_SIZE = 3
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SRC_SEQ_LEN = 800
TGT_SEQ_LEN = 800
MODEL_PATH = "./models/transformer.pt"  # Adjust this based on your saved model

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


def calculate_bleu(predictions, references):
    smoothing_function = SmoothingFunction().method4
    total_bleu = 0
    count = 0
    bleu=BLEU(effectiveorder=True)
    return  bleu.corpus_bleu(predictions,[references]).score
    # return sacrebleu.corpus_bleu(predictions,[references]).score
    # for pred, ref in zip(predictions, references):
    #     pred = [str(token) for token in pred]
    #     ref = [str(token) for token in ref]
    #     bleu = sentence_bleu([ref], pred, smoothing_function=smoothing_function)
    #     total_bleu += bleu
    #     count += 1
    # return total_bleu / count if count > 0 else 0.0


def evaluate(model, data_loader, device):
    model.eval()
    translations = []
    references = []
    
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            tgt_input = tgt_batch[:, :-1]
            src_mask, tgt_mask = create_masks(src_batch, tgt_input)

            encoder_output = model.encode(src_batch, src_mask)
            output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            output = model.project(output)
            predicted_tokens = output.argmax(dim=-1)

            translations.extend(predicted_tokens.tolist())
            references.extend(tgt_batch[:, 1:].tolist())

    return translations, references

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print("Loading test data...")
    with open("../../Dataset/en-hien/test.txt") as f:
        eng_test,hien_test = f.readlines().split("\t")

    test_dataset = TranslationDataset(eng_test, hien_test, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # Load the pre-trained model
    model = build_transformer(len(src_vocab), len(tgt_vocab), SRC_SEQ_LEN, TGT_SEQ_LEN, 768, 6, 8, 0.1, 3072)  # Use training vocabulary size
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    # Evaluate the model
    translations, references = evaluate(model, test_loader, device)

    # Calculate sentence-wise BLEU score and format the output
    sentence_bleu_scores = []
    count=0
    total_bleu=caluclate_bleu(translations,references)
    print(f"Total BLEU SCORE:{total_bleu}")
    output_file="transformers_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
            for src, true, pred in zip(datasets["test"]["source"], references, translations):
                f.write(f"Source: {src}\nTrue Translation: {references}\nGenerated Translation: {translations}\n\n")
    # for pred, ref in zip(translations, references):
    #     pred_sentence = ' '.join([str(token) for token in pred])  # Convert predicted tokens to sentence
    #     ref_sentence = ' '.join([str(token) for token in ref[1:]])  # Convert reference tokens to sentence, excluding <sos>
    #     bleu_score = calculate_bleu([pred], [ref[1:]])  # Calculate BLEU score
    #     sentence_bleu_scores.append(f"{eng_test[count].strip()}-{bleu_score:.4f}")
    #     count+=1
    #     total_bleu+=bleu_score
    # average_bleu = total_bleu / count if count > 0 else 0.0
    # Print formatted BLEU scores
    # with open("test_bleu.txt", 'w') as file:
    #     for score in sentence_bleu_scores:
    #         file.write(f"{score}\n")
        # file.write(f"Average BLEU Score: {average_bleu:.4f}\n")

if __name__ == "__main__":
    main()
