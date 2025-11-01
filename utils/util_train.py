import os
import torch
from torch_geometric.loader import DataLoader
import math
import time
from utils.util_plot import generate_dataframe, plot_bands, plot_loss, plot_test_loss
from config_file import palette, seedn 
torch.autograd.set_detect_anomaly(True)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_cumulative = 0.
    with torch.inference_mode():
        for d in dataloader:
            d.to(device)
            output = model(d)
        loss = loss_fn(output, d.y).cpu()
        loss_cumulative += loss.detach().item()
    return loss_cumulative / len(dataloader)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))


def train(model, optimizer, train_set, train_nums, test_set, 
          loss_fn, run_name, max_iter, scheduler, device, 
          batch_size, k_fold, factor=1000, conf_dict=None):
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    record_lines = []
    
    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
    else:
        print('Use torch.load to load the existing model: ' + run_name + '.torch')
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1
    
    train_sets = torch.utils.data.random_split(train_set, train_nums)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    for step in range(max_iter):
        k = step % k_fold
        train_loader = DataLoader(torch.utils.data.ConcatDataset(train_sets[:k] + train_sets[k+1:]), batch_size = batch_size, shuffle = True)
        valid_loader = DataLoader(train_sets[k], batch_size = batch_size)
        model.train()
        
        for i, data in enumerate(train_loader):
            start = time.time()
            data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y).cpu()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"num {i+1:4d} / {len(train_loader)}, loss = {loss}, train time = {time.time() - start:.2f} s, end = '\r'")
        end_time = time.time()
        wall = end_time - start_time
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            
            valid_avg_loss = evaluate(model, valid_loader, loss_fn, device)
            train_avg_loss = evaluate(model, train_loader, loss_fn, device)
            
            history.append({
                'step': s0 + step, 
                'batch':{
                    'loss': loss.item()
                }, 
                'valid': {
                    'loss': valid_avg_loss
                }, 
                'train': {
                    'loss': train_avg_loss
                },
            })
            
            results = {
                'history': history,
                'state': model.state_dict()
            }
            
            if conf_dict is not None:
                results['conf_dict'] = conf_dict
            
            print(f"Iteration {step+1:4d}    " + 
                  f"Train Loss: {train_avg_loss:.6f}    " + 
                  f"Valid Loss: {valid_avg_loss:.6f}    " + 
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
            
            save_name = f'./models/{run_name}'
            with open(save_name + '.torch', 'wb') as f:
                torch.save(results, f)
            
            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)
            plot_loss(history, save_name + '_loss')
            plot_test_loss(model, test_loader, loss_fn, device, save_name + '_loss_test')
            
            df_train = generate_dataframe(model, train_loader, loss_fn, device, factor)
            df_test = generate_dataframe(model, test_loader, loss_fn, device, factor)
            fit_train = plot_bands(df_train, header = save_name, title = 'train', n = 6, m = 2, palette = palette, formula = True, seed = seedn)
            fit_test = plot_bands(df_test, header = save_name, title = 'test', n = 6, m = 2, palette = palette, formula = True, seed = seedn)
        
        text_file = open(save_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()
        
        if scheduler is not None:
            scheduler.step()


def load_model(model_class, model_file, device):
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