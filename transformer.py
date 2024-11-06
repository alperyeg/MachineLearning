# Based on the tutorial on https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention mechanism computes the attention between each pair of
    positions in a sequence. It consists of multiple "attention heads"
    that capture different aspects of the input sequence.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "`d_model` must be dividable by `num_heads`"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculates the attention scores by first obtaining the similarity,
        between q and k, then applying the softmax to get the weights a.
        Finally, the attention is the matrix multiplication of the weights a and values V.
        """
        # calculate similarity using scaled dot product
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # apply now softmax to obtain weights a
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_head(self, x):
        """
        Reshape the input tensor into multiple heads.
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine the attention outputs from all heads
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Compute the multi-head self-attention,
        allowing the model to focus on some different aspects of the input sequence.
        """
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    The forward method applies the linear transformations and a ReLU activation function
    sequentially to compute the output. This process enables the model to consider
    the position of input elements while making predictions.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # self.fc1 = nn.Linear(d_model, d_ff)
        # self.fc2 = nn.Linear(d_ff, d_model)
        # self.relu = nn.ReLU()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # self.fc2(self.relu(self.fc1(x)))
        return self.model(x)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding is used to inject the position information of each token in the
    input sequence. It uses sine and cosine functions of different frequencies
    to generate the positional encoding.
    The class calculates sine and cosine values for even and odd indices, respectively,
    based on the scaling factor `div_term`.
    The forward method computes the positional encoding by adding the stored positional encoding
    values to the input tensor, allowing the model to capture the position
    information of the input sequence.
    """
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """
    An Encoder layer consists of a Multi-Head Attention layer,
    a Position-wise Feed-Forward layer, and two Layer Normalization layers.

    1.  The forward methods computes the encoder layer output by applying self-attention,
        adding the attention output to the input tensor, and normalizing the result.
    2. Then, it computes the position-wise feed-forward output, combines it with the
        normalized self-attention output, and normalizes the final result before returning
        the processed tensor.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # get attention
        attn_output = self.self_attn(x, x, x, mask)
        # add attention output to input and normalize, residual/skip connection
        x = self.norm1(x + self.dropout(attn_output))
        # get the position-wise feed-forward output of the normalized attention
        ff_output = self.feed_forward(x)
        # add the feed-forward output to the normalized attention and normalize again (residual/skip connection)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """
    A Decoder layer consists of a Multi-Head Attention layer, a Position-wise Feed-Forward layer,
    and three Layer Normalization layers.

    1. Calculate the masked self-attention output and add it to the input tensor,
        followed by dropout and layer normalization.
    2. Compute the cross-attention output between the decoder and encoder outputs,
        and add it to the normalized masked self-attention output, followed by dropout
        and layer normalization.
    3. Calculate the position-wise feed-forward output and combine it with the
        normalized cross-attention output, followed by dropout and layer normalization.
    4. Return the processed tensor.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                 dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_length)

        self.encode_layer = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decode_layer = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def generate_mask(src_tokens, tgt_tokens):
        src_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_tokens != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt_tokens.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.position_embedding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.position_embedding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encode_layer:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decode_layer:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

if __name__ == '__main__':
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                              num_heads, num_layers,
                              d_ff, max_seq_length, dropout)

    # Generate random data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    transformer.train()
    losses = []

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
        losses.append(loss.item())

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()