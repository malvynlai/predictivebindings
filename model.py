import torch 
import torch.nn as nn


class MoleculePredictor(nn.Module):
    def __init__(self, vocab_size=37, hidden_dim=64, inp_len=142, test=False):
        super().__init__()
        self.num_filters = 64
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0  
        )
        self.dropout_emb = nn.Dropout(0.2)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.num_filters*4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(self.num_filters*4, self.num_filters*2, kernel_size=3, padding='valid'),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(self.num_filters*2, self.num_filters, kernel_size=3, padding='valid'),
            nn.LeakyReLU(0.01)
        )
        
        self.flatten = nn.Flatten()
        flat_size = 2048  # 32 * 64 (num_filters)
        
        self.dense1 = nn.Sequential(
            nn.Dropout(0.3),
            WeightNormLinear(flat_size, 1024),
            nn.LeakyReLU(0.01)
        )
        
        self.dense2 = nn.Sequential(
            nn.Dropout(0.2),
            WeightNormLinear(flat_size + 1024, 1024),  
            nn.LeakyReLU(0.01)
        )
        
        self.dense3 = nn.Sequential(
            nn.Dropout(0.2),
            WeightNormLinear(flat_size + 2048, 1024),  
            nn.LeakyReLU(0.01)
        )
        self.output_layer = nn.Linear(1024, 3)



    def forward(self, x):
        x = self.embedding(x)  # (batch_size, inp_len, hidden_dim)
        x = self.dropout_emb(x)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, inp_len)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        hidden0 = self.flatten(x)  # batch_size x 2048
        hidden1 = self.dense1(hidden0)  # 2048 to 1024
        hidden2 = self.dense2(torch.cat([hidden0, hidden1], dim=1))  # (2048 + 1024) to 1024
        hidden3 = self.dense3(torch.cat([hidden0, hidden1, hidden2], dim=1))  # (2048 + 1024 + 1024) to 1024
        output = self.output_layer(hidden3)
        return output

class 