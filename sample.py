import time
import torch
from models.Predictor import Predictor
from utils.util_load import load_band_structure_data
from utils.util_data import generate_data_dict

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
    device = torch.device("cuda")
    model = Predictor().to(device)
    
    data = load_band_structure_data(DIR_CONFIG) # Load data
    data_dict = generate_data_dict(data, DATA_CONFIG) # Generate data dictionary

    for mpid, data in data_dict.items():
        print(f"Material ID: {mpid}")
        for key, value in data.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {value}")
        break
if __name__ == "__main__":
    main()