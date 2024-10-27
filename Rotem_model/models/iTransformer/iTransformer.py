import torch
import torch.nn as nn
import torch.nn.functional as F
from models.iTransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.iTransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer.layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, config):
        super(iTransformer, self).__init__()
        self.name = "iTransformer"

        # List of config items with _itrans suffix
        self.seq_len_itrans = config.sequence_length
        
        self.output_attention_itrans = config.output_attention_itrans
        d_model, output_size, fc_dropout= (
            
            config.d_model_itrans,
            config.num_labels,
            config.fc_dropout_transformer,
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            config.sequence_length, 
            config.d_model_itrans, 
            # config.embed_itrans, 
            # config.freq_itrans, 
            config.dropout_itrans
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=config.dropout_itrans,
                                    output_attention=config.output_attention_itrans), config.d_model_itrans, config.n_heads_itrans),
                    config.d_model_itrans,
                    config.d_ff_itrans,
                    dropout=config.dropout_itrans,
                ) for l in range(config.e_layers_itrans)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model_itrans)
        )
        fully_connected = []
        current_size = d_model
        while current_size//3 > output_size*2 :
            d_model = current_size//3
            fully_connected.append(nn.Linear(current_size, d_model))
            fully_connected.append(nn.ReLU())
            fully_connected.append(nn.Dropout(fc_dropout))

            current_size = d_model
        
        fully_connected.append(nn.Linear(current_size, output_size))
        self.fully_connected = nn.Sequential(*fully_connected)

        self.fc_sum =  nn.Linear(config.sequence_length, 1) 

    def forecast(self, x_enc, x_mark_enc=None):

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.fully_connected(enc_out)  
        dec_out = dec_out.permute(0,2,1)
        dec_out = self.fc_sum(dec_out)
        dec_out = dec_out.permute(0,2,1) # [batch,1,output=9]
        
        return dec_out


    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out # [B, L, D]