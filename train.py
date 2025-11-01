import os
import time
import torch

from sklearn.model_selection import train_test_split
from utils.util_load import load_band_structure_data
from utils.util_data import generate_data_dict
from utils.util_help import make_dict
from utils.util_train import train
from utils.util_loss import BandLoss
from models.Predictor import Predictor

from config_file import seedn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

file_name = os.path.basename(__file__)
print("File Name:", file_name)

model_dir = './models'
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

train_ratio = 0.9
batch_size = 1
k_fold = 5
max_iter = 200


node_dim = 118
edge_dim = 50
enc_layers = 4
dec_layers = 4
num_heads = 4
use_z = True # Whether to use one-hot
dropout = 0.1
d_model = 64

r_max = 8
descriptor = 'mass'
factor = 1000

loss_fn = BandLoss()
loss_fn_name = loss_fn.__class__.__name__
learning_rate = 5e-3
weight_decay = 5e-2
schedule_gamma = 0.96

conf_dict = make_dict([run_name, model_dir, data_dir, raw_dir, data_file, train_ratio, batch_size, k_fold, max_iter, 
                       node_dim, edge_dim, enc_layers, dec_layers, num_heads, use_z, dropout, d_model,
                       r_max, descriptor, factor,
                       loss_fn_name, learning_rate, weight_decay, schedule_gamma, seedn
                      ])
for k, v in conf_dict.items():
    print(f"{k}: {v}")


data = load_band_structure_data(data_dir, raw_dir, data_file)
data_dict = generate_data_dict(data = data, r_max = r_max, descriptor = descriptor, factor = factor)

num = len(data_dict)
train_nums = [int((num * train_ratio)//k_fold)] * k_fold
test_num = num - sum(train_nums)
idx_train, idx_test = train_test_split(range(num), test_size=test_num, random_state=seedn)
with open(f'./data/idx_{run_name}_train.txt', 'w') as f: 
    for idx in idx_train: f.write(f"{idx}\n")
with open(f'./data/idx_{run_name}_test.txt', 'w') as f: 
    for idx in idx_test: f.write(f"{idx}\n")

dataset = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
train_dataset, test_dataset = torch.utils.data.Subset(dataset, idx_train), torch.utils.data.Subset(dataset, idx_test)

model = Predictor(
    node_in_dim = node_dim, 
    d_model = d_model, 
    num_heads = num_heads,
    enc_layers = enc_layers,
    edge_in_dim = edge_dim,
    dec_layers = dec_layers,
    use_z = use_z,
    dropout = dropout
)

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=schedule_gamma)

train(model,  optimizer, train_dataset, train_nums, test_dataset, 
      loss_fn, run_name, max_iter, scheduler, device, batch_size, k_fold, 
      factor, conf_dict)