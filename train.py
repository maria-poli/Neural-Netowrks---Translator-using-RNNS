from dataload import Multi30kDataset, make_collator
from nltk.translate.bleu_score import sentence_bleu
from model import Encoder, Decoder, Seq2Seq
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import pickle
import torch
import math


DROPOUT = 0.1
EMB_DIM = 256
HID_DIM = 128
BATCH_SIZE = 256
TEACHER_FORCING_RATIO = 0.0

try:
    # Ensure data files are present and in the expected format
    train_df = pd.read_parquet("data/tokenized/train.parquet")
    valid_df = pd.read_parquet("data/tokenized/valid.parquet")
    
    # Initialize vocabularies (these should match the vocab.pkl files generated)
    with open("models/vocab_en.pkl", "rb") as f:
        vocab_en = pickle.load(f)
    with open("models/vocab_fr.pkl", "rb") as f:
        vocab_fr = pickle.load(f)

    # Initialize dataset
    train_data = Multi30kDataset(train_df, vocab_en, vocab_fr)
    valid_data = Multi30kDataset(valid_df, vocab_en, vocab_fr)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=make_collator(vocab_en, vocab_fr))
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=make_collator(vocab_en, vocab_fr))

except Exception as e:
    print("Error during dataset loading or initialization:", e)
    exit()

def train(model, iterator, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0

    for src, trg in iterator:
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    total_bleu_score = 0
    total_perplexity = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(model.device), trg.to(model.device)
            output = model(src, trg, 0)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            # Calculate BLEU score and Perplexity
            pred_tokens = output.argmax(1).cpu().numpy()
            trg_tokens = trg.cpu().numpy()
            bleu_score = sentence_bleu([trg_tokens], pred_tokens)
            perplexity = math.exp(loss.item())
            
            total_bleu_score += bleu_score
            total_perplexity += perplexity

    avg_bleu_score = total_bleu_score / len(iterator)
    avg_perplexity = total_perplexity / len(iterator)
    
    return epoch_loss / len(iterator), avg_bleu_score, avg_perplexity

def main():
    # Hyperparameters
    INPUT_DIM = len(vocab_en)
    OUTPUT_DIM = len(vocab_fr)
    N_EPOCHS = 10
    LEARNING_RATE = 0.001   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss(ignore_index=constants.PAD)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_en.token_to_index("<PAD>"))
    
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, TEACHER_FORCING_RATIO)
        valid_loss, bleu_score, perplexity = evaluate(model, valid_loader, criterion)
        
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val Loss: {valid_loss:.3f}, BLEU: {bleu_score:.3f}, Perplexity: {perplexity:.3f}')

if __name__ == '__main__':
    main()
