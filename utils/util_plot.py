import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter
from ase import Atom
import sklearn
import time
from config_file import palette, seedn, save_extension
from utils.util_help import chemical_symbols, sub
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def save_figure(fig, filename, title):
    """Save figure with white background and title."""
    fig.patch.set_facecolor('white')
    fig.suptitle(title, ha='center', y=1., fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    output_path = f"{filename}.{save_extension}"
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


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


def generate_dataframe(model, dataloader, loss_fn, device, factor=1000):
    rows = []
    model.eval()
    with torch.inference_mode():
        for d in dataloader:
            d = d.to(device)
            o = model(d)
            l = loss_fn(o, d.y).item()
            real = d.y.squeeze(0).detach().cpu().numpy() * factor
            pred = o.squeeze(0).detach().cpu().numpy() * factor
            rows.append({'id': d.id, 'name': d.symbol[0], 'loss': l, 'real': real, 'pred': pred, 'numb': int(d.numb)})
    return pd.DataFrame(rows)


def plot_atom_count_histogram(data, DIR_CONFIG, bins=None, figsize=(6, 5)):
    counts = [len(structure) for structure in data['structure']]
    if len(counts) == 0:
        return None
    if bins is None:
        bins = range(1, max(counts) + 2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts, bins=bins, color=palette[0])
    ax.set_xlabel('Atoms/Cell')
    ax.set_ylabel('Counts')
    os.makedirs(DIR_CONFIG['results_dir'], exist_ok=True)
    save_path = os.path.join(DIR_CONFIG['results_dir'], f"atoms_hist")
    save_figure(fig, save_path, title='Atoms per Cell')
    return counts


def plot_bands(df_in, header, title=None, n=5, m=1, num=3, lwidth=0.5, windowsize=(3, 2), palette=palette, formula=True, plot_real=True, save_lossx=False, seed=seedn):
    if seed is not None:
        np.random.seed(seed)
    fontsize = 10
    df_sorted = df_in.iloc[np.argsort(df_in['loss'])].reset_index(drop=True)
    losses = df_sorted['loss'].to_numpy()
    segments = [seg for seg in np.array_split(np.arange(len(df_sorted)), num) if len(seg) > 0]
    segment_count = len(segments)
    total_slots = segment_count * n * m
    choices = []
    for seg in segments:
        picks = np.random.choice(seg, size=n * m, replace=len(seg) < n * m)
        choices.extend(np.sort(picks))
    choices = np.array(choices[:total_slots])
    colors = palette[:segment_count]
    cols = np.repeat(colors, n * m)
    bounds = [losses[segments[0][0]]] + [losses[seg[-1]] for seg in segments]
    if len(losses) > 1:
        x_vals = np.linspace(losses.min(), losses.max(), 1000)
        p_vals = gaussian_kde(losses)(x_vals)
    else:
        x_vals = np.array([losses[0]])
        p_vals = np.ones_like(x_vals)
    if save_lossx:
        fig0, ax0 = plt.subplots(1, 1, figsize=(18, 2))
        ax0.plot(x_vals, p_vals, color='black')
        for idx, color in enumerate(colors):
            mask = (x_vals >= bounds[idx]) & (x_vals <= bounds[idx + 1])
            ax0.fill_between(x_vals, p_vals, where=mask, color=color, alpha=0.5)
        ax0.set_yticks([])
        ax0.set_xscale('log')
        ax0.tick_params(axis='x', which='major', labelsize=fontsize, rotation=90)
        ax0.tick_params(axis='x', which='minor', labelsize=fontsize, rotation=90)
        ax0.xaxis.set_minor_formatter(FormatStrFormatter("%.5f"))
        ax0.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        fig0.savefig(f"{header}_{title}_dist.{save_extension}")
        plt.close(fig0)
    rows = segment_count * m
    fig, axs = plt.subplots(rows, n + 1, figsize=((n + 1) * windowsize[1], rows * windowsize[0]), gridspec_kw={'width_ratios': [0.7] + [1] * n})
    axs = np.atleast_2d(axs)
    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
    axl = fig.add_subplot(gs[:, 0])
    axl.plot(p_vals, x_vals, color='black')
    for idx, color in enumerate(colors[::-1]):
        lower = bounds[len(bounds) - idx - 2]
        upper = bounds[len(bounds) - idx - 1]
        axl.fill_between([p_vals.min(), p_vals.max()], [lower, lower], [upper, upper], color=color, alpha=0.5)
    axl.invert_yaxis()
    axl.set_xticks([])
    axl.set_yscale('log')
    axl.tick_params(axis='y', which='major', labelsize=fontsize)
    axl.tick_params(axis='y', which='minor', labelsize=fontsize)
    axl.yaxis.set_minor_formatter(FormatStrFormatter("%.5f"))
    axl.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    axs = axs[:, 1:].ravel()
    id_list = []
    for idx, choice in enumerate(choices):
        ax = axs[idx]
        row = df_sorted.iloc[choice]
        real = np.asarray(row['real'])
        pred = np.asarray(row['pred'])
        q = pred.shape[0]
        x_axis = np.arange(q)
        if plot_real:
            if real.ndim == 1:
                ax.plot(x_axis, real, color='k', linewidth=lwidth * 0.8)
            else:
                for col in range(real.shape[1]):
                    ax.plot(x_axis, real[:, col], color='k', linewidth=lwidth * 0.6, alpha=0.6)
        if pred.ndim == 1:
            ax.plot(x_axis, pred, color=cols[idx], linewidth=lwidth)
        else:
            for col in range(pred.shape[1]):
                ax.plot(x_axis, pred[:, col], color=cols[idx], linewidth=lwidth)
        if formula:
            ax.set_title(simname(row['name']).translate(sub), fontsize=fontsize * 1.8)
        else:
            ax.set_title(row['id'], fontsize=fontsize * 1.8)
        id_list.append(row['id'])
        ax.tick_params(axis='y', which='major', labelsize=fontsize)
    save_figure(fig, f"{header}_{title}", title=title)


def compare_models(df1, df2, dir, labels=('Model1', 'Model2'), size=5, lw=3):
    color1 = palette[1]
    color2 = palette[3]
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
    
    file_name = os.path.join(dir, "model_comparison")
    save_figure(fig, file_name, title=f'{labels[0]} vs {labels[1]} Comparison')


def get_element_statistics(data_set):
    species = [Atom(Z).symbol for Z in range(1, 119)]
    species_dict = {k: [] for k in species}
    
    for i, data in enumerate(data_set):
        for specie in set(data.symbol):
            species_dict[specie].append(i)
    
    stats = pd.DataFrame({'symbol': species})
    stats['count'] = stats['symbol'].apply(lambda s: len(species_dict[s]))
    return stats


def plot_element_count_stack(data_set1, data_set2, dir, title):
    bar_colors=['#90BE6D', '#277DA1']
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
    
    fig.suptitle(title, ha='center', y=1.0, fontsize=20)
    fig.patch.set_facecolor('white')
    file_name = os.path.join(dir, f"element_count_stack")
    save_figure(fig, file_name, title)