import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from pytorch_tcn import TCN
import math 
# from AbsolutePositionalEncoding import AbsolutePositionalEncoding

class CNNLSTMModel(nn.Module):
    def __init__(self, config):
        super(CNNLSTMModel, self).__init__()
        input_dim, output_dim, sequence_length = config.input_size, config.num_labels, config.sequence_length

        self.name = "CNN_LSTMModel"
        self.config = config 
        self.temporal = False
        input_dim, lstm_hidden_size, lstm_num_layers, dropout,sequence_length, output_dim,cnn_hidden_size,kernel_size ,maxpoll_kernel_size= (
            config.input_size,
            config.lstm_hidden_size,
            config.lstm_num_layers,
            config.dropout,
            config.sequence_length,
            config.num_labels,
            config.cnn_hidden_size,
            config.cnn_kernel_size,
            config.maxpoll_kernel_size

        )
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_hidden_size, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size*2, kernel_size=kernel_size, padding='same')
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool1d(kernel_size=maxpoll_kernel_size, padding=0,stride=1)
        

        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(cnn_hidden_size)
        self.bn2 = nn.BatchNorm1d(cnn_hidden_size*2)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cnn_hidden_size*2, hidden_size=lstm_hidden_size,num_layers=lstm_num_layers,dropout=dropout ,batch_first=True,bidirectional=False)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, output_dim)
        
        # Activation function for LSTM
        self.relu = nn.ReLU()
        self.numofpool = 2 
        self.maxpoll_kernel_size = maxpoll_kernel_size
        # Sequence length for reshaping in forward pass
        self.sequence_length = sequence_length
        
    def forward(self, x):
        # Assume x is of shape (batch_size, sequence_length, input_dim)
        # Permute to shape (batch_size, input_dim, sequence_length) for conv layers
        x = x.permute(0, 2, 1)
        
        # Convolutional layers with PReLU activation
        x = self.bn1(self.prelu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(self.prelu(self.conv2(x)))
        x = self.pool(x)
        
        # Reshape for LSTM layer
        # We divide by 4 because we have two pooling layers with kernel_size=2
        x = x.permute(0, 2, 1)  # Back to (batch_size, sequence_length, input_dim)
        # x = x.reshape(-1, self.sequence_length // self.numofpool*self.maxpoll_kernel_size, self.config.cnn_hidden_size)
        
        # LSTM layer with ReLU activation
        x, (hn, cn) = self.lstm(x)
        x = self.relu(x)
        
        # Take the last time step output for a fully connected layer
        x = self.fc(x[:, -1:, :])
        
        return x




# class CNNLSTMModel(nn.Module):
#     def __init__(self, config):
#         super(CNNLSTMModel, self).__init__()
#         self.name = "CNN_LSTMModel"
#         self.config = config 
#         input_size, lstm_hidden_size, lstm_num_layers, dropout,sequence_length, output_size,cnn_hidden_size = (
#             config.input_size,
#             config.cnn_hidden_size,
#             config.lstm_num_layers,
#             config.dropout,
#             config.sequence_length,
#             config.num_labels,
#             config.cnn_hidden_size
#         )
#         self.temporal = False
#         self.lstm_num_layers = lstm_num_layers
#         self.lstm_hidden_size = lstm_hidden_size
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_hidden_size, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=cnn_hidden_size, out_channels=cnn_hidden_size, kernel_size=3, stride=1, padding=1)
#         self.drop = nn.Dropout1d(dropout)
#         self.relu = nn.ReLU()
#         self.batch_norm = nn.BatchNorm1d(lstm_hidden_size)

#         self.lstm = nn.LSTM(cnn_hidden_size, lstm_hidden_size, lstm_num_layers,proj_size = 6 ,batch_first=True, dropout=dropout, bidirectional=False,)
#         fully_connected = []
#         current_size = lstm_hidden_size*2
#         while current_size//3 > output_size*2:
#             hidden_size = current_size//3
#             fully_connected.append(nn.Linear(current_size, hidden_size))
#             fully_connected.append(nn.ReLU())
#             fully_connected.append(nn.Dropout(dropout))

#             current_size = hidden_size
        
#         fully_connected.append(nn.Linear(current_size, output_size))
#         self.fully_connected = nn.Sequential(*fully_connected)


#     def forward(self, x):


        
#         h0 = torch.zeros(self.lstm_num_layers , x.size(0), self.config.num_labels, dtype=torch.float32).to(x.device)
#         c0 = torch.zeros(self.lstm_num_layers , x.size(0), self.lstm_hidden_size, dtype=torch.float32).to(x.device)

#         # convlstm input [batch,featurs,sequence]
#         x = x.permute(0,2,1)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = x.permute(0,2,1) 
#         # LSTM gets [batch,sequence,feature]
#         out,_ = self.lstm(x, (h0.detach(), c0.detach()))

#         # out = out.permute(0,2,1)

#         # out = self.fully_connected(out[:, -1:, :])

#         return out[:, -1:, :]


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.name = "TransformerModel"
        self.config = config

        input_size, d_model, num_layers, output_size, fc_dropout,d_ff_transformer, n_head,head_dropout= (
            config.input_size,
            config.d_model_transformer,
            config.num_layers_transformer,
            config.num_labels,
            config.fc_dropout_transformer,
            config.d_ff_transformer,
            config.transformer_n_head,
            config.head_dropout_transformer
        )
        self.temporal = True
        # self.absenc = AbsolutePositionalEncoding()
        if d_model//n_head != 0:
            d_model = int(d_model/n_head)*n_head

        self.embedding = nn.Linear(input_size, d_model)
        self.temporal_embed = nn.Linear(1, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                    dim_feedforward=d_ff_transformer,activation=F.relu, batch_first=True,
                                    dropout=head_dropout),
            num_layers=num_layers , 
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
        # self.fully_connected =  nn.Linear(d_model, output_size) 


    def forward(self, x):
        #x:shape [batch,seq,feture]

        # x = self.embedding(x) + self.temporal_embed(x_time)
        x = self.embedding(x)

        x = self.transformer_encoder(x)

        x = self.fully_connected(x[:, -1:, :])
        return x

class DecompTransformerModel(nn.Module):
    def __init__(self, config):
        super(DecompTransformerModel, self).__init__()
        self.name = "DecompTransformerModel"
        self.config = config
        # add kernel size 
        input_size, d_model, num_layers, output_size, dropout = (
            config.input_size,
            config.d_model_transformer,
            config.num_layers_transformer,
            config.num_labels,
            config.dropout,
        )
        self.decomp = series_decomp(kernel_size=config.kernel_size)
        self.temporal = True

        # self.absenc = AbsolutePositionalEncoding()
        if d_model//self.config.n_head != 0:
            d_model = int(d_model/self.config.n_head)*self.config.n_head

        self.embedding = nn.Linear(input_size, d_model)
        self.temporal_embed = nn.Linear(1, d_model)


        self.transformer_encoder_season = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=self.config.n_head,dim_feedforward=self.config.d_ff,activation=F.leaky_relu, batch_first=True,
                                    dropout=config.head_dropout),
            num_layers=num_layers
        )

        self.transformer_encoder_trend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=self.config.n_head,dim_feedforward=self.config.d_ff,activation=F.leaky_relu, batch_first=True,
                                    dropout=config.head_dropout),
            num_layers=num_layers 
        )



        fully_connected = []
        current_size = d_model
        while current_size//3 > output_size*2 :
            d_model = current_size//3
            fully_connected.append(nn.Linear(current_size, d_model))
            fully_connected.append(nn.ReLU())
            fully_connected.append(nn.Dropout(dropout))

            current_size = d_model
        
        fully_connected.append(nn.Linear(current_size, output_size))
        self.fully_connected_season = nn.Sequential(*fully_connected)
        self.fully_connected_trend = nn.Sequential(*fully_connected)



    def forward(self, x,
                # x_time
                ):
        
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init, trend_init = seasonal_init, trend_init

        seasonal = self.embedding(seasonal_init) 
        trend = self.embedding(trend_init) 


        # season 
        # seasonal = seasonal.permute(1, 0, 2)  # Change the sequence length to be the first dimension
        seasonal = self.transformer_encoder_season(seasonal)
        # seasonal = seasonal.permute(1, 0, 2)  # Change it back to the original shape
        seasonal = self.fully_connected_season(seasonal[:, -1:, :])
        # trend
        # trend = trend.permute(1, 0, 2)  # Change the sequence length to be the first dimension
        trend = self.transformer_encoder_trend(trend)
        # trend = trend.permute(1, 0, 2)  # Change it back to the original shape
        trend = self.fully_connected_trend(trend[:, -1:, :])

        return seasonal + trend

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
    
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
    
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.name = "DLinear"
        self.config = configs
        self.seq_len = configs.sequence_length
        self.pred_len = configs.pred_len
        self.temporal = False

        # Decompsition Kernel Size
        
        self.decompsition = series_decomp(configs.kernel_size)
        self.individual = configs.individual
        self.channels = configs.input_size
        self.channels_out = configs.num_labels

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        # projection 
        self.attnproj = ChannelReductionAttentionModel(num_channels=self.channels,new_num_channels=self.channels_out,num_heads=self.config.dlinear_n_heads)
        self.fc_projection = nn.Linear(self.channels,self.channels_out)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
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

        x = x.permute(0,2,1) # to [Batch, Output length, Channel]
        if self.config.use_attnproj:
            return self.attnproj(x) 
        else:
            return self.fc_projection(x) 


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(
        #     nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv2 =nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.conv1 =nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self,config):
        super(TCN, self).__init__()
        input_size, output_size,num_channels, kernel_size, dropout = config.input_size,config.num_labels,config.num_channels,config.kernelsize_tcn,config.dropout
        self.name = "TCN"
        self.temporal = False

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # tcn input [batch,features,seq]
        x = x.permute(0,2,1)
        # x = x.flatten(1).unsqueeze(1)
        out = self.tcn(x)
        out = out.permute(0,2,1) # back to [batch,seq,feture]
        out = self.linear(out[:, -1:, :])
        
        return out
    



class CNN2D_LSTM(nn.Module):
    def __init__(self,config):
        super(CNN2D_LSTM, self).__init__()
        self.name = "Conv2DLSTMAttentionModel"
        self.temporal = False
        # Define 2D CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256*1*1, hidden_size=256, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.fc = nn.Linear(256, 9 ) # Define num_classes based on your task

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.pool1(F.relu(self.conv1(c_in)))
        c_out = self.pool2(F.relu(self.conv2(c_out)))
        c_out = self.pool3(F.relu(self.conv3(c_out)))
        c_out = self.pool4(F.relu(self.conv4(c_out)))
        c_out = c_out.view(batch_size, timesteps, -1)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(c_out)
        
        # Dropout layer
        out = self.dropout(lstm_out[:,-1,:])

        # Output layer
        out = self.fc(out)

        return out

class Conv2DLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super(Conv2DLSTMAttentionModel, self).__init__()
        self.name = "Conv2DLSTMAttentionModel"
        self.temporal = False
        # Configuration parameters
        input_dim, lstm_hidden_size, lstm_num_layers, dropout,sequence_length, output_dim,conv_hidden_sizes,kernel_size ,maxpoll_kernel_size,num_heads,maxpoll_layers= (
            config.input_size,
            config.conv2dlstm_hidden_size,
            config.conv2dlstm_num_layers,
            config.cnn2dlstm_dropout,
            config.sequence_length,
            config.num_labels,
            config.conv2d_hidden_sizes,
            config.cnn2d_kernel_size,
            config.cnn2dlstm_maxpoll_kernel_size,
            config.conv2d_n_heads,
            config.cnn2dlstm_maxpoll_layers


        )
        self.config = config
        # Convolutional layers
        self.convs = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(conv_hidden_sizes):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.ReLU()
            ]
            
            if i in maxpoll_layers:
                layers.append(nn.MaxPool2d(maxpoll_kernel_size))
            
            layers.append(nn.Dropout(dropout))

            self.convs.append(nn.Sequential(*layers))
            in_channels = out_channels  # Update in_channels for the next iteration

        feature_map_size = config.input_size//maxpoll_kernel_size**len(maxpoll_layers)
        lstm_hidden_size = conv_hidden_sizes[-1]*feature_map_size
        # LSTM layers
        self.lstm1 = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True, num_layers=lstm_num_layers)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=num_heads,dropout=dropout)

        # LSTM layer after attention
        self.lstm2 = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True)

        # Dense layer
        self.dense = nn.Linear(lstm_hidden_size, output_dim)

    def forward(self, x):
        # Input x should be of shape [batch, sequence, feature]

        # Permute x to shape [batch, feature, sequence] for Conv2D layers
        x = x.permute(0, 2, 1).unsqueeze(1)  # Adding a channel dimension

        # Apply convolutional layers
        for conv in self.convs:

            x = conv(x)

        # Reshape for LSTM layers
        # Removing the channel dimension and adjusting for the LSTM input
        
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, width, -1)

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
        x = self.dense(x[:, -1:, :])
        return x








class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        self.config = config
        self.name = "TimeSeriesTransformer"
        self.temporal = False

        # Linear layer for embedding: Input shape: [batch_size, seq_length, input_dim]
        self.embedding = nn.Linear(self.config.input_size, self.config.TST_d_model)

        # Positional encoding: Input shape: [batch_size, seq_length, d_model]
        self.positional_encoding = PositionalEncoding(self.config.TST_d_model, self.config.dropout, self.config.sequence_length)

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