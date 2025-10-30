import os
import torch
import torch.optim as optim
import accelerate
import numpy as np
import random

from utils.data_util import create_dataloader, MaterialDataset
from utils.train_util import param_count, visualize_frequencies
from utils.ema_util import ExponentialMovingAverage
from Models import PhononPredictor
from Models.loss import PhononLoss


def train():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    accelerator = accelerate.Accelerator()
    
    # 配置参数
    h5_path = os.path.join("Data", "materials_data.h5")
    epochs = 10000
    batch_size = 128
    learning_rate = 1e-4
    log_epoch = 40  # 每隔多少个epoch打印一次日志
    vis_epoch = 200  # 每隔多少个epoch进行一次可视化
    ema_decay = 0.999  # EMA衰减率
    
    # 创建模型保存和可视化目录
    model_dir = "Models"
    vis_dir = "visualization"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 模型参数
    node_feature_dim = 7
    edge_feature_dim = 1
    hidden_dim = 256
    
    model = PhononPredictor(
        node_dim=node_feature_dim,
        edge_dim=edge_feature_dim,
        hidden_dim=hidden_dim
    )
    
    dataset = MaterialDataset(h5_path)
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    if accelerator.is_main_process:
        print(f"Dataset loaded with {len(dataset)} materials.")
        param_count(model)
        
    criterion = PhononLoss(mse_weight=0.7, first_order_weight=0.2, second_order_weight=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 先准备模型、数据加载器、损失函数和优化器
    model, dataloader, criterion, optimizer = accelerator.prepare(
        model, dataloader, criterion, optimizer
    )
    
    # 在模型被accelerator.prepare()移至GPU后初始化EMA
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    
    # 跟踪最佳模型
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_mse, total_smooth1, total_smooth2 = 0, 0, 0, 0
        
        for batch_idx, (batch, material_ids) in enumerate(dataloader):
            graph_batch = batch['graph_batch']
            true_frequencies = batch['frequencies']   # 回到原始频率
            band_mask = batch['band_mask']
            
            optimizer.zero_grad()
            pred_frequencies = model(graph_batch)     # 直接预测频率 [B,120,256]
            loss, mse_loss, smooth_loss1, smooth_loss2 = criterion(pred_frequencies, true_frequencies, band_mask)
            accelerator.backward(loss)
            optimizer.step()
            ema.update()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_smooth1 += smooth_loss1.item()
            total_smooth2 += smooth_loss2.item()

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_smooth1 = total_smooth1 / len(dataloader)
        avg_smooth2 = total_smooth2 / len(dataloader)

        # 打印训练日志
        if (epoch + 1) % log_epoch == 0 and accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Avg Data Loss: {avg_mse:.4f} | Avg Smooth1: {avg_smooth1:.4f} | Avg Smooth2: {avg_smooth2:.4f}")

        # 可视化部分样本
        if (epoch + 1) % vis_epoch == 0 and accelerator.is_main_process:
            model.eval()
            ema.apply_shadow()
            
            # 随机选择一个批次进行可视化
            for batch_idx, (batch, material_ids) in enumerate(dataloader):
                if batch_idx > 0:
                    break
                    
                graph_batch = batch['graph_batch']
                true_frequencies = batch['frequencies']
                band_mask = batch['band_mask']
                
                best_model = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(best_model['model_state_dict'])
                with torch.no_grad():
                    pred_frequencies = unwrapped_model(graph_batch)
                
                # 选择前几个样本进行可视化
                num_samples = min(5, len(material_ids))
                for i in range(num_samples):
                    visualize_frequencies(
                        material_ids[i],
                        true_frequencies[i],
                        pred_frequencies[i],
                        band_mask[i],
                        save_dir=vis_dir,
                        epoch=epoch+1
                    )
                
                print(f"Visualized {num_samples} samples for epoch {epoch+1}")
                break
                
            ema.restore()
            model.train()
        
        # 保存最佳模型
        if avg_loss < best_loss and accelerator.is_main_process:
            best_loss = avg_loss
            ema.apply_shadow()

            save_path = os.path.join(model_dir, "best_model.pt")

            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            
            print(f"Saved best model to {save_path} with loss {best_loss:.4f}")
            ema.restore()
    
    accelerator.wait_for_everyone()
    print("Training completed!")

if __name__ == "__main__":
    train()