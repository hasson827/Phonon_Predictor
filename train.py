import time
import torch
from utils.util_load import load_band_structure_data
from utils.util_data import generate_data_dict
from utils.util_train import split_dataset, train
from utils.util_plot import plot_atom_count_histogram
from utils.util_loss import BandLoss
from models.Predictor import Predictor

DIR_CONFIG = dict(
    data_dir = './data', # Directory of all data
    raw_dir = './data/phonon', # Directory of raw data files
    data_file = 'DFPT_band_structure.pkl', # Processed data file
    results_dir = './results', # Directory to save results
    model_dir = './models', # Directory to save models and model architecture
    run_name = time.strftime('%y%m%d-%H%M%S', time.localtime()) # Name of the current run
)

DATA_CONFIG = dict(
    r_max = 8, # The cutoff distance for neighbor list
    descriptor = 'mass', # The node descriptor type
    factor = 1000, # The factor to scale the target values
    edge_K = 50, # The number of Gaussian basis functions for edge features
    edge_sigma = 0.5 # The width of Gaussian basis functions for edge features
)

MODEL_CONFIG = dict(
    node_in_dim = 118, # The input dimension of node features
    hidden_dim = 128, # The hidden dimension of the model
    num_encoder_layers = 3, # The number of layers in the Graph Encoder
    lmax = 1, # The maximum l value for spherical harmonics in the Graph Encoder
    extra_scalar_dim = 1, # The extra scalar dimension for node features in the Graph Encoder
    edge_scalar_dim = 50, # The scalar dimension for edge features in the Graph Encoder
    num_heads = 4, # The number of attention heads in the Graph Decoder
    num_decoder_layers = 3, # The number of layers in the Graph Decoder
    mlp_ratio = 2.0, # The MLP ratio in the Graph Decoder
    fourier_bands = 4, # The number of Fourier bands for q-point encoding in the Graph Decoder
    dropout = 0.1 # The dropout rate in both Encoder and Decoder
)

TRAIN_CONFIG = dict(
    train_ratio = 0.9, # The ratio of training data
    batch_size = 1, # The batch size
    k_fold = 5, # The number of folds for k-fold cross-validation
    max_iter = 1000, # The maximum number of training iterations
    learning_rate = 3e-3, # The learning rate
    weight_decay = 1e-2, # The weight decay for optimizer
    schedule_gamma = 0.95, # The gamma for learning rate scheduler
    use_ema = True, # Whether to enable EMA tracking
    ema_decay = 0.999, # The decay rate for EMA
    ema_start = 100, # The training step to start applying EMA weights
)

def main():
    # Basic Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    loss_fn = BandLoss()
    
    data = load_band_structure_data(DIR_CONFIG) # Load data
    data_dict = generate_data_dict(data, DATA_CONFIG) # Generate data dict
    plot_atom_count_histogram(data, DIR_CONFIG) # Plot atom count histogram
    train_dataset, test_dataset, train_nums = split_dataset(data_dict, TRAIN_CONFIG) # Split dataset
    
    # Model, Optimizer, Scheduler Setup
    model = Predictor(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TRAIN_CONFIG['schedule_gamma'])
    
    # main training & evaluation loop
    train(
        model = model, 
        optimizer = optimizer, 
        train_set = train_dataset, 
        train_nums = train_nums, 
        test_set = test_dataset,
        loss_fn = loss_fn, 
        scheduler = scheduler, 
        device = device, 
        DIR_CONFIG = DIR_CONFIG, 
        TRAIN_CONFIG = TRAIN_CONFIG, 
        DATA_CONFIG = DATA_CONFIG, 
        MODEL_CONFIG = MODEL_CONFIG
    )
    
    
if __name__ == '__main__':
    main()