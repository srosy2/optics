import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch import Tensor
import math
from torch.nn.utils.rnn import pad_sequence
from typing import List


PAD_IDX, BOS_IDX, EOS_IDX, LINS_IDX, AIR_IDX = range(5)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.fc = nn.Linear(13, emb_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.transform = sequential_transforms([tensor_transform, collate_fn])

    def forward(self, features: Tensor):
        tokens = self.transform(torch.tensor([AIR_IDX if x % 2 else LINS_IDX for x in range(
            features.shape[1])]).type(torch.long))
        embeddings = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        embeddings[:, 1:-1, :] += self.fc(features)
        return embeddings, tokens


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 actor: torch.nn.Module,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device='cpu'):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = actor
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,):
        src_emb, src_tokens = self.src_tok_emb(src)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb, tgt_tokens = self.tgt_tok_emb(trg)
        tgt_emb = self.positional_encoding(tgt_emb)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_tokens, tgt_tokens)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor):
        src_emb, src_tokens = self.src_tok_emb(src)
        src_mask = (torch.zeros(src_tokens.shape[1], src_tokens.shape[1])).type(torch.bool).to(self.device)
        return self.transformer.encoder(self.positional_encoding(src_emb), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor):
        tgt_emb, tgt_tokens = self.tgt_tok_emb(tgt)
        tgt_emb, tgt_tokens = tgt_emb[:, :-1], tgt_tokens[:, :-1]
        tgt_mask = (generate_square_subsequent_mask(tgt_tokens.shape[1])
                    .type(torch.bool)).to(self.device)
        return self.transformer.decoder(self.positional_encoding(
            tgt_emb), memory,
            tgt_mask)

    def greedy_decode(self, src, max_len, lfr=None):

        memory = self.encode(src)
        ys = torch.tensor([])
        memory = memory.to(self.device)
        for i in range(max_len - 1):
            out = self.decode(ys, memory)
            pred = self.generator.get_action(torch.cat([out[:, -1], lfr.view(1, -1)]))

            ys = torch.cat([ys, pred], dim=0)
            if i % 2 == 1 and pred[-1] > 0.5:
                break

        return ys, out

    # actual function to translate input sentence into target language
    def decode_without_target(self, src_sentence: torch.Tensor, lfr=None):
        src = self.transform(src_sentence).view(1, -1)

        tgt_pred, tgt_obs = self.greedy_decode(
            src, max_len=15, lfr=lfr)
        tgt_pred = tgt_pred.flatten()
        return tgt_pred, tgt_obs

    def decode_with_target(self, src, lfr, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        tgt_obs = self(src, tgt)
        tgt_obs = torch.cat([tgt_obs, torch.cat([lfr] * tgt_obs.shape[1]).view(1, tgt_obs.shape[1], -1)], dim=2)

        return tgt_obs


def collate_fn(src_batch, tgt_batch):
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = src == PAD_IDX
    tgt_padding_mask = tgt == PAD_IDX
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])
                     ))


if __name__ == '__main__':
    transformer_model = nn.Transformer(d_model=84, nhead=12, num_encoder_layers=4, num_decoder_layers=4,
                                       dim_feedforward=320,
                                       batch_first=True, dropout=0.)
    src = torch.rand((2, 10, 84))
    tgt = torch.rand((2, 10, 84))
    for i in [2, 3]:
        src_mask, tgt_mask = create_mask(src, tgt[:, :i, :])
        print(transformer_model(src, tgt[:, :i, :], src_mask, tgt_mask))
        print(transformer_model(src, tgt[:, :i, :]))
        # out_1 = transformer_model(src, tgt[:, :2, :])
        # out_2 = transformer_model(src, tgt[:, :2, :])
        # out_3 = transformer_model(src, tgt[:, :3, :])
    # print(src_mask)
    # print(tgt_mask)
    #
    # print(torch.sum(torch.abs(out_1 - out_2)))
    # print(torch.sum(torch.abs(out_1 - out_3[:, :2, :])))
