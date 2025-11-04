import os
import math
import time
import torch
from typing import Tuple
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from utils.util_plot import plot_element_count_stack, compare_models, generate_dataframe, plot_loss, plot_bands
from config_file import palette, seedn 
torch.autograd.set_detect_anomaly(True)


def split_dataset(data_dict: dict, TRAIN_CONFIG: dict) -> Tuple[Dataset, Dataset, list]:
    train_ratio = TRAIN_CONFIG['train_ratio']
    k_fold = TRAIN_CONFIG['k_fold']
    
    num = len(data_dict)
    train_nums = [int((num * train_ratio) // k_fold)] * k_fold
    test_num = num - sum(train_nums)
    idx_train, idx_test = train_test_split(range(num), test_size=test_num, random_state=seedn)
    dataset = Subset(list(data_dict.values()), range(len(data_dict)))
    train_dataset, test_dataset = Subset(dataset, idx_train), Subset(dataset, idx_test)
    return train_dataset, test_dataset, train_nums


def count_param(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.inference_mode():
        for d in dataloader:
            d = d.to(device)
            output = model(d)
            loss = loss_fn(output, d.y)
            loss_sum += float(loss.detach().cpu())
            n += 1
    return loss_sum / max(n, 1)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))


def train(model, optimizer, train_set, train_nums, test_set, 
        loss_fn, scheduler, device,  DIR_CONFIG, TRAIN_CONFIG, DATA_CONFIG):
    num_params = count_param(model)
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    record_lines = []
    best_valid = float('inf')
    
    # Configuration parameters
    model_dir = DIR_CONFIG['model_dir']
    results_dir = DIR_CONFIG['results_dir']
    run_name = DIR_CONFIG['run_name']
    
    k_fold = TRAIN_CONFIG['k_fold']
    batch_size = TRAIN_CONFIG['batch_size']
    max_iter = TRAIN_CONFIG['max_iter']
    
    factor = DATA_CONFIG['factor']
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, run_name + '.torch')
    
    # Plot the element count stack
    plot_element_count_stack(train_set, test_set, results_dir, 
                             title='Element Count in Train and Test Sets')
    results = {}
    history = []
    s0 = 0
    
    
    train_sets = torch.utils.data.random_split(train_set, train_nums)
    
    fold_loaders = []    
    for k in range(k_fold):
        train_concat = ConcatDataset(train_sets[:k] + train_sets[k+1:])
        train_loader = DataLoader(train_concat, batch_size = batch_size, shuffle = True)
        valid_loader = DataLoader(train_sets[k], batch_size = batch_size, shuffle = False)
        fold_loaders.append((train_loader, valid_loader))
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    for step in range(max_iter):
        k = step % k_fold
        train_loader, valid_loader = fold_loaders[k]
        model.train()
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        results_save_prefix = os.path.join(run_dir, f"step_{step}")

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            
            valid_avg_loss = evaluate(model, valid_loader, loss_fn, device)
            train_avg_loss = evaluate(model, train_loader, loss_fn, device)
            
            history.append({
                'step': s0 + step, 
                'batch':{'loss': float(loss.detach().cpu())}, 
                'valid': {'loss': valid_avg_loss}, 
                'train': {'loss': train_avg_loss},
            })
            
            if valid_avg_loss < best_valid:
                best_valid = valid_avg_loss
                results = {'history': history, 'state': model.state_dict()}
                with open(weights_path, 'wb') as f:
                    torch.save(results, f)
                df_train = generate_dataframe(model, train_loader, loss_fn, device, factor)
                df_test = generate_dataframe(model, test_loader, loss_fn, device, factor)
                plot_bands(df_train, header = results_save_prefix, title = 'train', n = 6, m = 2, palette = palette, formula = True, seed = seedn)
                plot_bands(df_test, header = results_save_prefix, title = 'test', n = 6, m = 2, palette = palette, formula = True, seed = seedn)

            record_lines.append(f"{step}\t{train_avg_loss:.20f}\t{valid_avg_loss:.20f}")
            metrics_path = os.path.join(run_dir, f"metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Number of parameters: {num_params}\nstep\ttrain_loss\tvalid_loss\n")
                for line in record_lines:
                    f.write(line + "\n")
            
            plot_loss(history, os.path.join(run_dir, "loss_curve"))
    compare_models(df_train, df_test, dir=run_dir, labels=('Train', 'Test'), size=5, lw=3)


def load_model(model_class, model_file, device):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} does not exist.")

    print(f"Loading model from {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        checkpoint = torch.load(model_file)
        
        conf_dict = checkpoint.get('conf_dict', checkpoint)
        
        model = model_class(**conf_dict)
        model.load_state_dict(checkpoint['state'])
        model.to(device)
        
        history = checkpoint.get('history', [])
        s0 = history[-1]['step'] + 1 if history else 0
        
        return model, conf_dict, history, s0
    else:
        raise FileNotFoundError(f"Model file {model_file} does not exist.")