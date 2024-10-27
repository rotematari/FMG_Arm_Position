from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import PositionsDataset
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train
from models.SOFTS.models.SOFTS import Model as SOFTS
from iTransformer import iTransformer,iTransformer2D
if __name__ == '__main__':
    # Load the config file
    config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(config["seed"])
    device = set_device()

    # Load the dataset
    trainDataset = PositionsDataset(config,mode='train')
    testDataset = PositionsDataset(config,mode='test')
    # Create the dataloader
    trainDataloader = DataLoader(trainDataset, batch_size=config["batch_size"], shuffle=True,drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=config["batch_size"], shuffle=False,drop_last=True)
    # Create the model
    # model = SOFTS(config)
    model = iTransformer(
    num_variates = config["iTransformer_num_variates"], # number of variates in the dataset
    lookback_len = config["sequence_length"],          # lookback length of the model or the sequence length 
    dim = config["iTransformer_dim"],                          # model dimensions
    depth = config["iTransformer_depth"],                          # depth
    heads = config["iTransformer_heads"],                          # attention heads
    dim_head = config["iTransformer_dim_head"],                      # head dimension
    pred_length = config["iTransformer_pred_length"],     # can be one prediction, or many
    num_tokens_per_variate = config["iTransformer_num_tokens_per_variate"],         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    use_reversible_instance_norm = config["iTransformer_use_reversible_instance_norm"] # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    )
    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Create the loss function
    # criterion = torch.nn.MSELoss()
    # Train the model
    train(config,model,optimizer, trainDataloader,testDataloader,trainDataset.scalar, device='cpu',wandb_run=None)