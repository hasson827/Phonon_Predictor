import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

def param_count(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {params / 1e6:.2f}M")
    return

def visualize_frequencies(material_id, true_frequencies, pred_frequencies, band_mask=None, save_dir="visualization", epoch=None):
    """
    可视化真实频率和预测频率的对比图
    
    Args:
        material_id: 材料ID
        true_frequencies: 真实频率 (shape: [bands, points])
        pred_frequencies: 预测频率 (shape: [bands, points])
        band_mask: 频带掩码，指示哪些频带是有效的
        save_dir: 保存可视化图像的目录
        epoch: 当前训练的epoch
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保输入是numpy数组
    if isinstance(true_frequencies, torch.Tensor):
        true_frequencies = true_frequencies.detach().cpu().numpy()
    if isinstance(pred_frequencies, torch.Tensor):
        pred_frequencies = pred_frequencies.detach().cpu().numpy()
    if band_mask is not None and isinstance(band_mask, torch.Tensor):
        band_mask = band_mask.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    x_axis = np.arange(true_frequencies.shape[1])
    
    # 确定有效的频带数量
    num_bands = true_frequencies.shape[0]
    if band_mask is not None:
        num_bands = int(band_mask.sum())
    
    # 绘制每个频带
    for i in range(num_bands):
        if band_mask is None or band_mask[i] > 0.5:  # 仅绘制有效频带
            plt.plot(x_axis, true_frequencies[i], '-', color='blue', alpha=0.7, linewidth=1)
            plt.plot(x_axis, pred_frequencies[i], '--', color='red', alpha=0.7, linewidth=1)
    
    # 添加图例和标题
    plt.legend(['True Frequencies', 'Predicted Frequencies'])
    plt.title(f'Phonon Frequencies - {material_id}')
    plt.xlabel('q-point index')
    plt.ylabel('Frequency (THz)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像，文件名格式为：epoch_n_materialid.png
    if epoch is not None:
        filename = f"epoch_{epoch}_{material_id}.png"
    else:
        filename = f"{material_id}.png"
        
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")