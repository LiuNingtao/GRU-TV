import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v)
        
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc(attention)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Transformer(nn.Module):
    def __init__(self, num_classes, num_variables, num_layers=4, d_model=128, num_heads=8, dff=512, dropout_rate=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(num_variables, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)])
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=1)
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding to the input
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, mask)
        
        # Global average pooling
        x = self.global_avg_pooling(x.permute(0, 2, 1)).squeeze(-1)
        
        # Classification layer
        x = self.fc(x)
        output_cls = torch.sigmoid(x[:, :-1])
        if x.shape[1] > 1 or x.shape[1] == 1: 
            output_reg = x[:, -1]
        else:
            output_reg = None
        return output_cls, output_reg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == '__main__':
    import numpy as np

    # Generate random sequence length ranging from 5 to 100

    # Number of variables
    num_variables = 10

    # Example usage:
    num_classes = 2
    model = Transformer(num_classes, num_variables)
    for i in range(10):
        sequence_length = np.random.randint(5, 13754)
        demo_input = np.random.rand(2, sequence_length, num_variables)
        demo_mask = np.random.rand(2, sequence_length, num_variables)
        demo_mask[demo_mask<0.5] = 0
        demo_mask[demo_mask!=0] = 1
        print("Demo input shape:", demo_input.shape)
        output = model(torch.tensor(demo_input).float())
        print(output)
