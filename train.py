import os
import time
import torch

from sklearn.model_selection import train_test_split
from utils.util_load import load_band_structure_data
from utils.util_data import generate_data_dict
from utils.util_help import make_dict

file_name = os.path.basename(__file__)
print("File Name:", file_name)

seedn = 42
data_dir = './data'
raw_dir = './data/phonon'
data_file = 'DFPT_band_structure.pkl'
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

train_ratio = 0.9
batch_size = 1
k_fold = 5

r_max = 4
descriptor = 'mass'
factor = 1000

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