import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from pytorch_tcn import TCN
import math 
from iTransformer import iTransformer,iTransformer2D
import math
from iTransformer import iTransformer, iTransformer2D

class iTransformerModel(nn.Module):
    def __init__(self, config):
        super(iTransformerModel, self).__init__()
        self.name = "iTransformerModel"
        self.config = config

        input_size = config.input_size
        d_model = config.iTransformer_dim
        output_size = config.num_labels
        fc_dropout = config.fc_dropout_transformer

        # Projection layer to map input features to model dimensions
        self.embedding = nn.Linear(input_size, d_model)

        # Initialize the iTransformer model
        self.iTransformer = iTransformer(
            num_variates=config.iTransformer_num_variates,
            lookback_len=config.sequence_length,
            dim=d_model,
            depth=config.iTransformer_depth,
            heads=config.iTransformer_heads,
            dim_head=config.iTransformer_dim_head,
            pred_length=config.iTransformer_pred_length,
            num_tokens_per_variate=config.iTransformer_num_tokens_per_variate,
            use_reversible_instance_norm=config.iTransformer_use_reversible_instance_norm
        )

        # Fully connected layers
        fully_connected = []
        current_size = input_size
        while current_size // 3 > output_size * 2:
            next_size = current_size // 3
            fully_connected.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            current_size = next_size

        fully_connected.append(nn.Linear(current_size, output_size))
        self.fully_connected = nn.Sequential(*fully_connected)

        # Final linear layer to reduce sequence length dimension
        self.fc_sum = nn.Linear(config.sequence_length, 1)

    def forward(self, x, mask=None):
        # x shape: [batch_size, sequence_length, input_size]
        # x = self.embedding(x)  # Projection layer
        x = self.iTransformer(x)  # iTransformer encoder
        x = self.fully_connected(x[1])  # Fully connected layers
        # x = x.permute(0, 2, 1)  # Permute to [batch_size, output_size, sequence_length]
        # x = self.fc_sum(x)  # Reduce sequence length dimension
        # x = x.permute(0, 2, 1)  # Permute back to [batch_size, 1, output_size]
        return x

class iTransformer2DModel(nn.Module):
    def __init__(self, config):
        super(iTransformer2DModel, self).__init__()
        self.name = "iTransformer2DModel"
        self.config = config

        input_size = config.input_size
        d_model = config.iTransformer_dim
        output_size = config.num_labels
        fc_dropout = config.fc_dropout_transformer

        # Projection layer to map input features to model dimensions
        self.embedding = nn.Linear(input_size, d_model)

        # Initialize the iTransformer2D model
        self.iTransformer2D = iTransformer2D(
            num_variates=config.iTransformer_num_variates,
            num_time_tokens=config.iTransformer_num_time_tokens,
            lookback_len=config.sequence_length,
            dim=d_model,
            depth=config.iTransformer_depth,
            heads=config.iTransformer_heads,
            dim_head=config.iTransformer_dim_head,
            pred_length=config.iTransformer_pred_length,
            use_reversible_instance_norm=config.iTransformer_use_reversible_instance_norm
        )

        # Fully connected layers
        fully_connected = []
        current_size = input_size
        while current_size // 3 > output_size * 2:
            next_size = current_size // 3
            fully_connected.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            current_size = next_size

        fully_connected.append(nn.Linear(current_size, output_size))
        self.fully_connected = nn.Sequential(*fully_connected)

        # Final linear layer to reduce sequence length dimension
        self.fc_sum = nn.Linear(config.sequence_length, 1)

    def forward(self, x, mask=None):
        # x shape: [batch_size, sequence_length, input_size]
        x = self.iTransformer2D(x)  # iTransformer2D encoder
        x = self.fully_connected(x[1])  # Fully connected layers

        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, config):
        super(CNNLSTMModel, self).__init__()
        self.config = config
        self.name = "CNNLSTMModel"

        # Extract hyperparameters from config with CNNLSTMModel prefix
        num_filters = config.get('CNNLSTMModel_num_filters', [32, 64])
        kernel_size = config.get('CNNLSTMModel_kernel_size', 3)
        lstm_hidden_size = config.get('CNNLSTMModel_lstm_hidden_size', 128)
        lstm_num_layers = config.get('CNNLSTMModel_lstm_num_layers', 2)
        dropout = config.get('CNNLSTMModel_dropout', 0.5)
        use_batchnorm = config.get('CNNLSTMModel_use_batchnorm', True)
        input_channels = config.get('input_size', 32)
        num_classes = config.get('output_size', 9)
        sequence_length = config.get('sequence_length', 128)

        num_conv_layers = len(num_filters)
        assert len(num_filters) == num_conv_layers, "Length of num_filters must match num_conv_layers"

        layers = []
        in_channels = input_channels  # Input channels in time series data (like the input size)

        # Build convolutional layers
        for i in range(num_conv_layers):
            out_channels = num_filters[i]
            
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            layers.append(conv_layer)

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))

            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = out_channels
            sequence_length = sequence_length // 2  # Adjust sequence length after each pooling

        self.conv_layers = nn.Sequential(*layers)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)

        # Calculate the size after the CNN layers and pooling
        lstm_input_size = sequence_length * in_channels*2

        # Fully connected layers for downscaling
        fully_connected = []
        current_size = lstm_input_size
        while current_size // 3 > num_classes * 2:
            next_size = current_size // 3
            fully_connected.append(nn.Linear(current_size, next_size))
            fully_connected.append(nn.ReLU())
            fully_connected.append(nn.Dropout(dropout))
            current_size = next_size
        
        fully_connected.append(nn.Linear(current_size, num_classes))
        self.fully_connected = nn.Sequential(*fully_connected)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        # Transpose to [batch_size, input_size, sequence_length] for conv layers
        x = x.permute(0, 2, 1)

        # Pass through CNN layers
        x = self.conv_layers(x)

        # Permute to [batch_size, sequence_length, features] for LSTM
        x = x.permute(0, 2, 1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Flatten for fully connected layer
        x = x.reshape(x.size(0), -1)

        # Apply dropout and fully connected layer
        x = self.dropout(x)
        x = self.fully_connected(x)

        return x


class CNN1D(nn.Module):
    def __init__(self, config):
        super(CNN1D, self).__init__()
        self.config = config
        self.name = "CNN1D"
        # Extract hyperparameters from config
        
        num_filters = config.get('num_filters', [32, 64])
        kernel_size = config.get('kernel_sizes', 3)
        use_batchnorm = config.get('use_batchnorm', True)
        dropout = config.get('dropout', 0.5)
        input_channels = config.get('input_channels', 32)
        num_classes = config.get('output_size', 9)  
        sequence_length = config.get('sequence_length', 100)
        num_conv_layers = len(num_filters)
        
        assert len(num_filters) == num_conv_layers, "Length of num_filters must match num_conv_layers"


        layers = []
        in_channels = input_channels  # This is input_size in time series data

        # Build convolutional layers
        for i in range(num_conv_layers):
            out_channels = num_filters[i]
            kernel_size = kernel_size

            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            layers.append(conv_layer)

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))

            layers.append(nn.ReLU())

            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            in_channels = out_channels

            # Update sequence_length after pooling
            sequence_length = sequence_length // 2

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of the flattened features after convolutional layers
        flattened_size = in_channels * sequence_length

        # Fully connected layer
        self.fc = nn.Linear(flattened_size, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        # Transpose to [batch_size, input_size, sequence_length]
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x



class FullyConnectedNetwork(nn.Module):
    def __init__(self, config):
        super(FullyConnectedNetwork, self).__init__()
        self.name = "FullyConnectedNetwork"
        # Extract hyperparameters from config
        input_size = config["input_size"]*config["sequence_length"]
        
        output_size = config["output_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]

        layers = []
        in_size = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # input [batch,sequence,featurs]
        batch_size = x.size(0)
        x = x.view(batch_size, -1) #[batch,sequence*featurs]

        return self.network(x)

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


# class TransformerModel(nn.Module):
#     def __init__(self, config):
#         super(TransformerModel, self).__init__()
#         self.name = "TransformerModel"
#         self.config = config

#         input_size, d_model, num_layers, output_size, fc_dropout,d_ff_transformer, n_head,head_dropout= (
#             config.input_size,
#             config.d_model_transformer,
#             config.num_layers_transformer,
#             config.num_labels,
#             config.fc_dropout_transformer,
#             config.d_ff_transformer,
#             config.transformer_n_head,
#             config.head_dropout_transformer
#         )
#         self.temporal = True

#         if d_model % n_head != 0:
#             d_model = (d_model // n_head) * n_head

#         self.embedding = nn.Linear(input_size, d_model)

#         # Positional encoding: Input shape: [batch_size, seq_length, d_model]
#         self.positional_encoding = PositionalEncoding(self.config.d_model_transformer, self.config.dropout)

#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
#                                     dim_feedforward=d_ff_transformer,activation=F.gelu, batch_first=True,
#                                     dropout=head_dropout),
#             num_layers=num_layers
#         )

#         fully_connected = []
#         current_size = d_model
#         while current_size//2 > output_size*2 :
#             d_model = current_size//2
#             fully_connected.append(nn.Linear(current_size, d_model))
#             fully_connected.append(nn.ReLU())
#             fully_connected.append(nn.Dropout(fc_dropout))

#             current_size = d_model
        
#         fully_connected.append(nn.Linear(current_size, output_size))
#         self.fully_connected = nn.Sequential(*fully_connected)

#         self.fc_sum =  nn.Linear(self.config.sequence_length, 1) 

#     def forward(self, x, mask=None):

#         x = self.embedding(x) * math.sqrt(self.config.d_model_transformer)

#         x = self.positional_encoding(x)

#         x = self.transformer_encoder(x,mask=mask)

#         x = self.fully_connected(x) # B, 
#         x = x.permute(0,2,1)
#         x = self.fc_sum(x)
#         x = x.permute(0,2,1) # [batch,1,output=9]
#         return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ChannelReductionAttentionModel(nn.Module):
    def __init__(self, num_channels, num_heads, new_num_channels):
        super(ChannelReductionAttentionModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.linear = nn.Linear(num_channels, new_num_channels)

    def forward(self, x):
        # input x shape [Batch, seq_len , Channel]
        x = x.permute(1, 0, 2)  # Change shape for multihead attention to x shape: [seq_len, batch, num_channels]
        attn_output, _ = self.multihead_attn(x, x, x)
        # attn_output shape: [seq_len, batch, num_channels]
        transformed_output = self.linear(attn_output)
        # transformed_output shape: [seq_len, batch, new_num_channels]
        return transformed_output.permute(1, 0, 2)  # Change back to [batch, seq_len, new_num_channels]

# Example usage


class AttentionWeightedAverage(nn.Module):
    def __init__(self, seq_len, num_channels):
        super(AttentionWeightedAverage, self).__init__()
        self.attention_weights = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, num_channels]
        weights = self.attention_weights(x)  # -> [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)  # Normalize weights
        weighted_avg = torch.sum(weights * x, dim=1, keepdim=True)  # [batch, 1, num_channels]
        return weighted_avg
    

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.name = "DLinear"
        self.config = configs
        self.sequence_length = configs["sequence_length"]
        self.pred_len = configs.get("pred_len", 1)
        self.temporal = False

        # Decompsition Kernel Size
        self.decompsition = series_decomp(configs["Dlinear_kernel_size"])
        self.individual = configs["individual"]
        self.channels = configs["input_size"]
        self.channels_out = configs["output_size"]

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.sequence_length,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.sequence_length,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.sequence_length,self.pred_len)
            self.Linear_Trend = nn.Linear(self.sequence_length,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        self.fc_projection = nn.Linear(self.channels,self.channels_out)
        # self.fc_sum = nn.Linear(self.sequence_length,1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x: [Batch, sequence_length, input_size]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        x = x.permute(0,2,1) # to [Batch,sequence_length, Channel]

        return self.fc_projection(x)[:,-1,:] 


class Conv2DLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super(Conv2DLSTMAttentionModel, self).__init__()
        self.name = "Conv2DLSTMAttentionModel"
        self.temporal = False

        # Extract configuration parameters using config.get()
        input_dim = config.get('input_size', 32)  # Default input dimension
        sequence_length = config.get('sequence_length', 100)  # Default sequence length
        output_dim = config.get('output_size', 9)  # Default output dimension
        lstm_num_layers = config.get('conv2dlstm_num_layers', 1)
        dropout = config.get('cnn2dlstm_dropout', 0.5)
        conv_hidden_sizes = config.get('conv2d_hidden_sizes', [32, 64])
        kernel_size = config.get('cnn2d_kernel_size', 3)
        maxpool_kernel_size = config.get('cnn2dlstm_maxpool_kernel_size', 2)
        num_heads = config.get('conv2d_n_heads', 4)
        maxpool_layers = config.get('cnn2dlstm_maxpool_layers', [0])  # Layers at which to apply max pooling

        self.config = config


        # Convolutional layers
        self.convs = nn.ModuleList()
        in_channels = 1

        for i, out_channels in enumerate(conv_hidden_sizes):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size,stride=1, padding="same"),
                nn.ReLU()
            ]
            
            if i in maxpool_layers:
                layers.append(nn.MaxPool2d(maxpool_kernel_size))
            
            layers.append(nn.Dropout(dropout))

            self.convs.append(nn.Sequential(*layers))
            in_channels = out_channels  # Update in_channels for the next iteration

        # Calculate output dimensions after convolutions and pooling
        h_out = input_dim
        w_out = sequence_length
        for i in range(len(conv_hidden_sizes)):
            if i in maxpool_layers:
                h_out = h_out // maxpool_kernel_size
                w_out = w_out // maxpool_kernel_size

        # LSTM dimensions
        lstm_input_size = conv_hidden_sizes[-1] * h_out
        lstm_hidden_size = lstm_input_size//2  # You can adjust this as needed

        # feature_map_size = config.input_size//maxpoll_kernel_size**len(maxpoll_layers)
        # lstm_hidden_size = conv_hidden_sizes[-1]*feature_map_size
        # LSTM layers
        self.lstm1 = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True, num_layers=lstm_num_layers,bidirectional=True)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size*2, num_heads=num_heads,dropout=dropout)
        lstm_2_hidden_size = lstm_hidden_size
        # LSTM layer after attention
        self.lstm2 = nn.LSTM(lstm_hidden_size*2, lstm_2_hidden_size, batch_first=True,bidirectional=True)

        # Dense layer
        self.dense = nn.Linear(lstm_2_hidden_size*2, output_dim)
        self.sum = nn.Linear(w_out, 1)

    def forward(self, x):
        # Input x should be of shape [batch, sequence, feature]
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape: [batch_size, 1, input_dim, sequence_length]


        # Apply convolutional layers
        for conv in self.convs:

            x = conv(x)

        # x shape after convolutions: [batch_size, channels, h_out, w_out]
        batch_size, channels, h_out, w_out = x.size()
        # Reshape for LSTM
        x = x.view(batch_size, w_out, channels * h_out)  # Shape: [batch_size, seq_len, features]

        # Apply first LSTM layer
        # Input to LSTM is of shape [batch, seq_len, features]
        x, _ = self.lstm1(x)

        # Prepare data for MultiheadAttention
        # Convert to shape [seq_len, batch, features] for attention
        x = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(x, x, x)

        # Revert back to shape [batch, seq_len, features] after attention
        x = attn_output.permute(1, 0, 2)

        # Apply second LSTM layer
        x, _ = self.lstm2(x)

        # Apply dense layer
        # Using the output of the last time step
        x = self.dense(x)
        x = x.permute(0,2,1)
        x = self.sum(x)
        x = x.permute(0,2,1)
        return x[:,-1,:]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        self.config = config
        self.name = "TimeSeriesTransformer"
        self.temporal = False

        # Linear layer for embedding: Input shape: [batch_size, seq_length, input_dim]
        self.embedding = nn.Linear(self.config.input_size, self.config.TST_d_model)

        # Positional encoding: Input shape: [batch_size, seq_length, d_model]
        self.positional_encoding = PositionalEncoding(self.config.TST_d_model, self.config.dropout)

        # Transformer: Input shape: [seq_length, batch_size, d_model] for both src and tgt
        self.transformer = nn.Transformer(d_model=self.config.TST_d_model, 
                                          nhead=self.config.TST_n_head, 
                                          num_encoder_layers=self.config.TST_encoder_layers,
                                          num_decoder_layers=self.config.TST_decoder_layers,
                                          dim_feedforward=self.config.TST_d_ff, 
                                          dropout=self.config.dropout)

        # Output linear layer: Input shape: [seq_length, batch_size, d_model]
        self.out = nn.Linear(self.config.TST_d_model, self.config.num_labels)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if tgt is None:
            tgt = src[:,-1,:]



        src = self.embedding(src) * math.sqrt(self.config.TST_d_model)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * math.sqrt(self.config.TST_d_model)
        tgt = self.positional_encoding(tgt)

        # Reshape src and tgt to [seq_length, batch_size, d_model]
        src = src.transpose(0, 1)  # Reshape from [batch_size, seq_length, d_model] to [seq_length, batch_size, d_model]
        tgt = tgt.transpose(0, 1)  # Same reshaping for tgt
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        output = self.out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar