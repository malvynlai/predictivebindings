import torch 
import torch.nn as nn


class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class CNN1d(nn.Module):
    def __init__(self, vocab_size=37, hidden_dim=64, inp_len=142, test=False):
        super().__init__()
        self.num_filters = 64
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0  
        )
        self.dropout_emb = nn.Dropout(0.1)
        
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
            nn.Dropout(0.1),
            WeightNormLinear(flat_size, 1024),
            nn.LeakyReLU(0.01)
        )
        
        self.dense2 = nn.Sequential(
            nn.Dropout(0.1),
            WeightNormLinear(flat_size + 1024, 1024),  
            nn.LeakyReLU(0.01)
        )
        
        self.dense3 = nn.Sequential(
            nn.Dropout(0.1),
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


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.silu = SiLU()
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.fc(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x


class CNN1dWithGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=19, stride=1, padding='same')
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same')
        self.conv3 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding='same')
        self.silu = SiLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.gru = nn.GRU(192, 128, bidirectional=True, batch_first=True)
        self.mlp1 = MLPBlock(192 + 256, 1024)  # 256 from bidirectional GRU (128*2)
        self.mlp2 = MLPBlock(1024, 1024)
        self.mlp3 = MLPBlock(1024, 512)
        self.output = nn.Linear(512, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        x = self.silu(self.conv3(x))
        conv_out = self.global_pool(x).squeeze(-1)
        gru_in = x.transpose(1, 2)  
        gru_out, _ = self.gru(gru_in)
        gru_out = self.global_pool(gru_out.transpose(1, 2)).squeeze(-1)
        x = torch.cat([conv_out, gru_out], dim=1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class CNN1dWithoutGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=19, stride=1, padding='same')
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same')
        self.conv3 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding='same')
        self.silu = SiLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp1 = MLPBlock(192, 1024)
        self.mlp2 = MLPBlock(1024, 1024)
        self.mlp3 = MLPBlock(1024, 512)
        self.output = nn.Linear(512, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  
        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))
        x = self.silu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    