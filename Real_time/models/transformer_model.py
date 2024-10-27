import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.name = "TransformerModel"
        self.config = config

        # Config parameters using dictionary access
        input_size = config["input_size"]
        d_model = config["d_model_transformer"]
        num_layers = config["num_layers_transformer"]
        output_size = config["output_size"]
        fc_dropout = config["fc_dropout_transformer"]
        d_ff_transformer = config["d_ff_transformer"]
        n_head = config["transformer_n_head"]
        head_dropout = config["head_dropout_transformer"]
        self.activation = config["activation"]  # Add activation function to the config
        use_learnable_pe = config["use_learnable_positional_encoding"]

        # Ensure d_model is divisible by n_head
        if d_model % n_head != 0:
            d_model = (d_model // n_head) * n_head

        self.temporal = True

        # Embedding layer
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding: choose between learnable or fixed
        if use_learnable_pe:
            self.positional_encoding = LearnablePositionalEncoding(d_model, config["sequence_length"], config["dropout"])
        else:
            self.positional_encoding = PositionalEncoding(d_model, config["dropout"])

        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff_transformer,
                activation=self.get_activation(),
                batch_first=True,
                dropout=head_dropout
            ),
            num_layers=num_layers
        )

        # # Fully connected layers for downscaling
        # fully_connected = []
        # current_size = d_model
        # fully_connected.append(self.get_activation_layer())
        # fully_connected.append(nn.Dropout(fc_dropout))
        # while current_size // 3 > output_size * 3:
        #     next_size = current_size // 3
        #     fully_connected.append(nn.Linear(current_size, next_size))
        #     fully_connected.append(self.get_activation_layer())
        #     fully_connected.append(nn.Dropout(fc_dropout))
        #     current_size = next_size
        
        # fully_connected.append(nn.Linear(current_size, output_size))
        # self.fully_connected = nn.Sequential(*fully_connected)
                # Final layer to sum across sequence dimension
        # self.fc_sum = nn.Linear(config["sequence_length"], 1)
        self.wrist_fc = self.make_fc(d_model,fc_dropout,3)
        self.wrist_fc_sum = nn.Linear(config["sequence_length"], 1)
        self.elbow_fc = self.make_fc(d_model,fc_dropout,3)
        self.elbow_fc_sum = nn.Linear(config["sequence_length"], 1)
        self.unsupervised_fc = self.make_fc(d_model,fc_dropout,32)
        self.unsupervised_fc_sum = nn.Linear(config["sequence_length"], 1)
    def get_activation(self):
        """Return the activation function as per config."""
        if self.activation == 'relu':
            return 'relu'
        elif self.activation == 'gelu':
            return 'gelu'
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def make_fc(self,d_model,fc_dropout,output_size):
                # Fully connected layers for downscaling
        fully_connected = []
        current_size = d_model
        fully_connected.append(self.get_activation_layer())
        fully_connected.append(nn.Dropout(fc_dropout))
        while current_size // 3 > output_size * 3:
            next_size = current_size // 3
            fully_connected.append(nn.Linear(current_size, next_size))
            fully_connected.append(self.get_activation_layer())
            fully_connected.append(nn.Dropout(fc_dropout))
            current_size = next_size
        
        fully_connected.append(nn.Linear(current_size, output_size))
        full_fully_connected = nn.Sequential(*fully_connected)

        return full_fully_connected
    def get_activation_layer(self):
        """Return the corresponding PyTorch activation function."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x, mask=None):
        # Input embedding and scaling
        x = self.embedding(x) * math.sqrt(self.config["d_model_transformer"])
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)

        fmg = self.unsupervised_fc(x)

        elbow = self.elbow_fc(x)
        wrist = self.wrist_fc(x)

        # Reshape and sum over sequence length
        elbow = elbow.permute(0, 2, 1)
        elbow = self.elbow_fc_sum(elbow)
        elbow = elbow.permute(0, 2, 1)  # Output: [batch, 1, output_size]
        elbow = elbow[:,-1,:]
        # Reshape and sum over sequence length
        wrist = wrist.permute(0, 2, 1)
        wrist = self.wrist_fc_sum(wrist)
        wrist = wrist.permute(0, 2, 1)  # Output: [batch, 1, output_size]
        wrist = wrist[:,-1,:]
        # # Fully connected layers
        # x = self.fully_connected(x)

        # # Reshape and sum over sequence length
        # x = x.permute(0, 2, 1)
        # x = self.fc_sum(x)
        # x = x.permute(0, 2, 1)  # Output: [batch, 1, output_size]
        # x = x[:,-1,:]
        return elbow,wrist,fmg
    

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for transformer models."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer models."""
    def __init__(self, d_model, sequence_length, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return self.dropout(x)

