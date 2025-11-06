import os
import time
import torch
from typing import Tuple
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from utils.util_plot import plot_element_count_stack, compare_models, generate_dataframe, plot_loss, plot_bands
from utils.util_ema import EMA
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


def train(model, optimizer, train_set, train_nums, test_set, 
    loss_fn, scheduler, device,  DIR_CONFIG, TRAIN_CONFIG, DATA_CONFIG, MODEL_CONFIG):
    
    # Configuration parameters
    use_ema = TRAIN_CONFIG.get('use_ema', False)
    ema = EMA(model, decay=TRAIN_CONFIG.get('ema_decay', 0.999)) if use_ema else None
    ema_start = TRAIN_CONFIG.get('ema_start', 0)
    
    model_dir = DIR_CONFIG.get('model_dir', './models')
    results_dir = DIR_CONFIG.get('results_dir', './results')
    run_name = DIR_CONFIG.get('run_name', time.strftime('%y%m%d-%H%M%S', time.localtime()))

    k_fold = TRAIN_CONFIG.get('k_fold', 5)
    batch_size = TRAIN_CONFIG.get('batch_size', 1)
    max_iter = TRAIN_CONFIG.get('max_iter', 200)

    factor = DATA_CONFIG.get('factor', 1000)
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, run_name + '.torch')
    
    # Plot the element count stack
    plot_element_count_stack(train_set, test_set, results_dir, title='Element Count')
    
    # Initialize training variables
    num_params = count_param(model)
    model.to(device)
    record_lines = []
    best_valid = float('inf')
    results = {}
    history = []
    s0 = 0
    df_train, df_test = None, None
    
    # Prepare k-fold data loaders
    fold_loaders = []  
    train_sets = torch.utils.data.random_split(train_set, train_nums)
    for k in range(k_fold):
        train_concat = ConcatDataset(train_sets[:k] + train_sets[k+1:])
        train_loader = DataLoader(train_concat, batch_size = batch_size, shuffle = True)
        valid_loader = DataLoader(train_sets[k], batch_size = batch_size, shuffle = False)
        fold_loaders.append((train_loader, valid_loader))
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    # Save configuration to file
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("DATA_CONFIG:\n")
        for key, value in DATA_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write("\nMODEL_CONFIG:\n")
        for key, value in MODEL_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTRAIN_CONFIG:\n")
        for key, value in TRAIN_CONFIG.items():
            f.write(f"{key}: {value}\n")
    
    # Training loop
    for step in range(max_iter):
        # Initialize for this step
        k = step % k_fold
        train_loader, valid_loader = fold_loaders[k]
        results_save_prefix = os.path.join(run_dir, f"step_{step}")
        
        # Main training step
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None and step >= ema_start:
                ema.update()
        scheduler.step()

        # Evaluate and log metrics
        if step % 5 == 0:  
            # Apply EMA if enabled
            ema_applied = False
            if ema is not None and step >= ema_start:
                ema.apply_shadow()
                ema_applied = True
            
            # Evaluate on validation and training sets for logging and loss plotting
            valid_avg_loss = evaluate(model, valid_loader, loss_fn, device)
            train_avg_loss = evaluate(model, train_loader, loss_fn, device)
            
            history.append({
                'step': s0 + step,
                'batch': {'loss': float(loss.detach().cpu())},
                'valid': {'loss': valid_avg_loss},
                'train': {'loss': train_avg_loss},
            })
            
            plot_loss(history, os.path.join(run_dir, "loss_curve"))
            record_lines.append(f"{step}\t{train_avg_loss:.20f}\t{valid_avg_loss:.20f}")
            metrics_path = os.path.join(run_dir, f"metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Model Size: {num_params/1e6:.2f} MB\nstep\ttrain_loss\tvalid_loss\n")
                for line in record_lines:
                    f.write(line + "\n")
            
            # Save best model and generate plots
            if valid_avg_loss < best_valid:
                best_valid = valid_avg_loss
                results = {'history': history, 'state': model.state_dict()}
                with open(weights_path, 'wb') as f:
                    torch.save(results, f)
                df_train = generate_dataframe(model, train_loader, loss_fn, device, factor)
                df_test = generate_dataframe(model, test_loader, loss_fn, device, factor)
                plot_bands(df_train, header = results_save_prefix, title = 'train', n = 6, m = 2, palette = palette, seed = seedn)
                plot_bands(df_test, header = results_save_prefix, title = 'test', n = 6, m = 2, palette = palette, seed = seedn)

            # Restore EMA weights if applied
            if ema_applied:
                ema.restore()
    # Final model comparison plots (Best Model)
    if df_train is not None and df_test is not None:
        compare_models(df_train, df_test, dir=run_dir, labels=('Train', 'Test'), size=5, lw=3)