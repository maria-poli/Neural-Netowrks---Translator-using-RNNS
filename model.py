import torch.nn.functional as F
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        # self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True) #for 2.1
        self.rnn = nn.LSTM(emb_dim, hid_dim // 2, batch_first=True, bidirectional=True) #for2.2
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        emb = self.dropout(self.emb(input))
        outputs, (hid, cell) = self.rnn(emb)
        
        # adapt hid and cell from the bidirectional lstm layer
        hid = torch.cat((hid[-2, :, :], hid[-1, :, :]), dim=1).unsqueeze(0) #for2.2
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0) #for2.2

        return hid, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output, hid, cell):
        emb = self.dropout(self.emb(output))
        output, (hid, cell) = self.lstm(emb, (hid, cell))
        prediction = self.fc_out(output.squeeze(1))  
        return prediction, hid, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, output, teacher_forcing_ratio=0.5):
        batch_size = output.size(0)
        target_length = output.size(1)
        target_vocab_size = self.decoder.fc_out.out_features
        
        results = torch.zeros(batch_size, target_length, target_vocab_size).to(input.device)
        
        hidden, cell = self.encoder(input)
        input = output[:, 0]  # First input is <SOS>
        
        for t in range(1, target_length):
            result, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            results[:, t, :] = result
            p = torch.rand(1).item() 
            if teacher_forcing_ratio != 0.0:
                if p <= teacher_forcing_ratio:
                    input = output[:, t] 
            else: output.argmax(1)
        
        return results
