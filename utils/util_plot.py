import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter
from ase import Atom
from copy import copy
import sklearn
import time
from tqdm import tqdm
from config_file import palette, seedn, save_extension
from utils.util_help import chemical_symbols, sub


def save_figure(fig, filename, title = None):
    fig.patch.set_facecolor('white')
    if title:
        fig.suptitle(title, ha='center', y=1., fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(f"{filename}.{save_extension}")


def plot_loss(history, filename):
    steps = [d['step'] for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, loss_train, 'o-', label='Training', color=palette[3])
    ax.plot(steps, loss_valid, 'o-', label='Validation', color=palette[1])
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    save_figure(fig, filename, title="Loss Curve")


def plot_test_loss(model, dataloader, loss_fn, device, filename):
    model.eval()
    model.to(device)
    loss_test = []
    
    with torch.inference_mode():
        for data in dataloader:
            data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y).cpu()
            loss_test.append(loss.item())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(loss_test), label=f"Testing Loss: {np.mean(loss_test)}", color=palette[3])
    ax.set_ylabel("Loss")
    ax.legend()
    save_figure(fig, filename, title="Test Loss")


def simname(symbol):
    count = 0
    prev = ''
    name = ''
    for s in symbol:
        if s != prev:
            if name != '':
                name += str(count)
            name += s
            prev = s
            count = 1
        else:
            count += 1
    name += str(count)
    return name


def loss_dist_general(axl, df, num, palette, tile_losses, fontsize, axis='y'):
    loss_min, loss_max = df['loss'].min(), df['loss'].max()
    points = np.linspace(loss_min, loss_max, 5000)
    kde = gaussian_kde(list(df['loss']))
    density = kde.pdf(points)

    if axis == 'y':
        axl.plot(density, points, color='black')
    else:
        axl.plot(points, density, color='black')
    
    cols = palette[:num]
    cols_rev = copy(cols)
    cols_rev.reverse()
    quantiles = list(tile_losses)[::-1] + [0]
    
    for i in range(len(quantiles) - 1):
        if axis == 'y':
            axl.fill_between([density.min(), density.max()], y1=[quantiles[i], quantiles[i]], y2=[quantiles[i+1], quantiles[i+1]], color=cols_rev[i], lw=0, alpha=0.5)
        else:
            axl.fill_between(points, density, where=(points >= quantiles[i+1]) & (points <= quantiles[i]), color=cols_rev[i], lw=0, alpha=0.5)
    
    if axis == 'y':
        axl.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axl.invert_yaxis()
        axl.set_xticks([])
        axl.set_yscale('log')
        axl.tick_params(axis='y', which='major', labelsize=fontsize)
        axl.tick_params(axis='y', which='minor', labelsize=fontsize)
        axl.yaxis.set_minor_formatter(FormatStrFormatter("%.5f"))
        axl.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    else:
        axl.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axl.set_yticks([])
        axl.set_xscale('log')
        axl.tick_params(axis='x', which='major', labelsize=fontsize, rotation=90)
        axl.tick_params(axis='x', which='minor', labelsize=fontsize, rotation=90)
        axl.xaxis.set_minor_formatter(FormatStrFormatter("%.5f"))
        axl.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        
        axl.spines['top'].set_visible(False)
        axl.spines['right'].set_visible(False)
        axl.spines['left'].set_visible(False)
    return axl, cols


def generate_dataframe(model, dataloader, loss_fn, device, factor=1000):
    df = pd.DataFrame(columns=['id', 'name', 'loss', 'real', 'pred', 'time', 'numb'])
    with torch.no_grad():
        for d in tqdm(dataloader):
            try:
                d.to(device)
                start_time = time.time()
                output = model(d)
                run_time = time.time() - start_time
                if loss_fn is not None:
                    loss = loss_fn(output, d.y).cpu().item()
                else: 
                    loss = 0

                real = d.y.cpu().numpy() * factor
                pred = output.cpu().numpy() * factor
                rrr = {'id': d.id, 'name': d.symbol, 'loss': loss, 'real': list(real), 'pred': list(np.array([pred])), 'time': run_time, 'numb': d.numb.cpu()}
                df0 = pd.DataFrame(data = rrr)
                df = pd.concat([df, df0], ignore_index=True)
            except Exception as e:
                print(e, d.id)
                continue
    return df

def plot_general(df_in, header, title=None, n=5, m=1, num=3, lwidth=0.5, windowsize=(3, 2), palette=palette, formula=True, plot_func=None, plot_real=True, save_lossx=False, seed=seedn):
    if seed is not None:
        np.random.seed(seed)
    
    fontsize = 10
    df_sorted = df_in.iloc[np.argsort(df_in['loss'])].reset_index(drop = True)
    tiles = np.arange(1, num + 1)/num
    tile_losses = np.quantile(df_sorted['loss'], tiles)
    idx_q = [0] + [np.argmin(np.abs(df_sorted['loss'] - tile_loss)) for tile_loss in tile_losses]
    replace = True if len(df_sorted) < n * m * num else False
    s = np.concatenate([np.sort(np.random.choice(np.arange(idx_q[k], idx_q[k+1], 1), size=m * n, replace=replace)) for k in range(num)])
    
    if save_lossx:
        fig0, axl0 = plt.subplots(1, 1, figsize=(18, 2))
        axl0, cols0 = loss_dist_general(axl0, df_sorted, num, palette, tile_losses, fontsize, axis='x')
        fig0.savefig(f"{header}_{title}_dist.{save_extension}")
        
    fig, axs = plt.subplots(num * m, n + 1, figsize=((n + 1) * windowsize[1], num * m * windowsize[0]), gridspec_kw={'width_ratios': [0.7] + [1] * n})
    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
        
    axl = fig.add_subplot(gs[:, 0])
    axl, cols = loss_dist_general(axl, df_sorted, num, palette, tile_losses, fontsize, axis='y')

    cols = np.repeat(cols, n * m)
    axs = axs[:, 1:].ravel()
    
    id_list = []
    for k in range(num * m * n):
        ax = axs[k]
        i = s[k]
        real, pred = df_sorted.iloc[i]['real'], df_sorted.iloc[i]['pred']
        plot_func(ax, real, pred, cols[k], lwidth, plot_real=plot_real)

        if formula:
            ax.set_title(simname(df_sorted.iloc[i]['name']).translate(sub), fontsize=fontsize * 1.8)
        else:
            ax.set_title(df_sorted.iloc[i]['id'], fontsize=fontsize * 1.8)
        id_list.append(df_sorted.iloc[i]['id'])
        ax.tick_params(axis='y', which='major', labelsize=fontsize)

    save_figure(fig, f"{header}_{title}", title=title)
    print(id_list)
    return fig


def plot_band(ax, real, pred, color, lwidth, qticks=None, plot_real=True, ylabel=False):
    xpts = pred.shape[0]
    if plot_real and real is not None:
        ax.plot(range(xpts), real, color='k', linewidth=lwidth * 0.8)
    ax.plot(range(xpts), pred, color=color, linewidth=lwidth)
    if qticks is not None:
        ax.set_xticks(range(xpts))
        qticks = [f"${txt}$" if not '$' in txt and len(txt) > 0 else txt for txt in qticks]
        ax.set_xticklabels(qticks, fontsize=10)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
    if ylabel:
        ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$')


def plot_bands(df_in, header, title=None, n=5, m=1, lwidth=0.5, windowsize=(3, 2), palette=palette, formula=True, plot_real=True, save_lossx=False, seed=seedn):
    return plot_general(df_in=df_in, header=header, title=title, 
                        n=n, m=m, num=3, lwidth=lwidth, windowsize=windowsize, 
                        palette=palette, formula=formula, plot_func=plot_band, 
                        plot_real=plot_real, save_lossx=save_lossx, seed=seed)


def compare_models(df1, df2, header, color1, color2, labels=('Model1', 'Model2'), size=5, lw=3, r2=False):
    re_out1 = np.concatenate([df1.iloc[i]['real'].flatten() for i in range(len(df1))])
    pr_out1 = np.concatenate([df1.iloc[i]['pred'].flatten() for i in range(len(df1))])
    re_out2 = np.concatenate([df2.iloc[i]['real'].flatten() for i in range(len(df2))])
    pr_out2 = np.concatenate([df2.iloc[i]['pred'].flatten() for i in range(len(df2))])

    min_val = min(re_out1.min(), pr_out1.min(), re_out2.min(), pr_out2.min())
    max_val = max(re_out1.max(), pr_out1.max(), re_out2.max(), pr_out2.max())
    width = max_val - min_val

    min_x1_loss, max_x1_loss = df1['loss'].min(), df1['loss'].max()
    x1_loss = np.linspace(min_x1_loss, max_x1_loss, 500)
    kde1 = gaussian_kde(df1['loss'])
    p1 = kde1.pdf(x1_loss)

    min_x2_loss, max_x2_loss = df2['loss'].min(), df2['loss'].max()
    x2_loss = np.linspace(min_x2_loss, max_x2_loss, 500)
    kde2 = gaussian_kde(df2['loss'])
    p2 = kde2.pdf(x2_loss)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.8))  # Two subplots in one row

    ax1 = axs[0]
    ax1.plot([min_val - 0.01 * width, max_val + 0.01 * width], [min_val - 0.01 * width, max_val + 0.01 * width], color='k')
    ax1.set_xlim(min_val - 0.01 * width, max_val + 0.01 * width)
    ax1.set_ylim(min_val - 0.01 * width, max_val + 0.01 * width)
    ax1.scatter(re_out1, pr_out1, s=size, marker='.', color=color1, label=labels[0])
    ax1.scatter(re_out2, pr_out2, s=size, marker='.', color=color2, label=labels[1])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel('True $\omega$ [$cm^{-1}$]', fontsize=14)
    ax1.set_ylabel('Predicted $\omega$ [$cm^{-1}$]', fontsize=14)
    ax1.legend()
    
    if r2:
        R2_1 = sklearn.metrics.r2_score(y_true=re_out1, y_pred=pr_out1)
        R2_2 = sklearn.metrics.r2_score(y_true=re_out2, y_pred=pr_out2)
        ax1.set_title(f"$R^2 [1]: {R2_1:.3f}$\t$ [2]: {R2_2:.3f}$", fontsize=14)

    ax2 = axs[1]
    ax2.plot(x1_loss, p1, color=color1, label=labels[0], lw=lw)
    ax2.plot(x2_loss, p2, color=color2, label=labels[1], lw=lw)
    ax2.set_xscale('log')
    
    min_y1_loss, max_y1_loss = np.min(p1), np.max(p1)
    min_y2_loss, max_y2_loss = np.min(p2), np.max(p2)
    min_x_loss, max_x_loss = min([min_x1_loss, min_x2_loss]), max([max_x1_loss, max_x2_loss])
    min_y_loss, max_y_loss = min([min_y1_loss, min_y2_loss]), max([max_y1_loss, max_y2_loss])
    width_x_loss = max_x_loss - min_x_loss
    width_y_loss = max_y_loss - min_y_loss
    ax2.set_xlim(min_x_loss - 0.08 * width_x_loss, max_x_loss + 0.08 * width_x_loss)
    ax2.set_ylim(min_y_loss - 0.08 * width_y_loss, max_y_loss + 0.08 * width_y_loss)

    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_xlabel('Loss', fontsize=14)
    ax2.set_ylabel('Probability density', fontsize=14)
    ax2.legend()
    
    fig.suptitle(f'{labels[0]} vs {labels[1]} Comparison', ha='center', y=0.98, fontsize=16)
    save_figure(fig, f"{header}_model_comparison", title=None)


def get_element_statistics(data_set):
    species = [Atom(Z).symbol for Z in range(1, 119)]
    species_dict = {k: [] for k in species}
    
    for i, data in enumerate(data_set):
        for specie in set(data.symbol):
            species_dict[specie].append(i)
    
    stats = pd.DataFrame({'symbol': species})
    stats['count'] = stats['symbol'].apply(lambda s: len(species_dict[s]))
    return stats


def plot_element_count_stack(data_set1, data_set2, header=None, title=None, bar_colors=['#90BE6D', '#277DA1'], save_fig=False):
    stats1 = get_element_statistics(data_set1)
    stats2 = get_element_statistics(data_set2)
    
    common_elems = stats1[stats1['count'] > 0]['symbol'].tolist() + stats2[stats2['count'] > 0]['symbol'].tolist()
    common_elems = sorted(set(common_elems))
    common_elems = sorted(common_elems, key=lambda x: chemical_symbols.index(x))
    
    stats1 = stats1[stats1['symbol'].isin(common_elems)].set_index('symbol').reindex(common_elems)
    stats2 = stats2[stats2['symbol'].isin(common_elems)].set_index('symbol').reindex(common_elems)
    
    rows = 2
    fig, axs = plt.subplots(rows, 1, figsize=(27, 10 * rows))
    bar_max = max(stats1['count'].max(), stats2['count'].max())
    
    for i, ax in enumerate(axs):
        col_range = range(i * (len(stats1) // rows), (i + 1) * (len(stats1) // rows))
        ax.bar(col_range, stats1['count'].iloc[col_range], width=0.6, color=bar_colors[0], label='Train')
        ax.bar(col_range, stats2['count'].iloc[col_range], bottom=stats1['count'].iloc[col_range], width=0.6, color=bar_colors[1], label='Test')
        
        ax.set_xticks(col_range)
        ax.set_xticklabels(stats1.index[col_range], fontsize=27)
        ax.set_ylim(0, bar_max * 1.2)
        ax.tick_params(axis='y', which='major', labelsize=23)
        ax.set_ylabel('Counts', fontsize=24)
        ax.legend()
    
    if title:
        fig.suptitle(title, ha='center', y=1.0, fontsize=20)
    
    if save_fig:
        fig.patch.set_facecolor('white')
        fig.savefig(f'{header}_element_count_{title}.{save_extension}')
        save_figure(fig, f'{header}_element_count_{title}', title=title)