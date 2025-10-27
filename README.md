# Phonon Predictor

Graph-based model to predict phonon band frequencies from crystal structures with data pulled from the Materials Project. The pipeline builds a graph via Voronoi neighbors, trains a GNN to output band-by-q frequencies, and logs visualizations and the best checkpoint.

## Key features
- Automated data download from Materials Project and HDF5 storage
- Pytorch Geometric-based graph construction (Voronoi neighbors + simple atomic features)
- Resampling to a fixed [bands=120, q-points=256] grid and masking for variable bands
- A compact GNN predictor with multi-layer message passing and global pooling
- Training with Accelerate, EMA, masked losses, and periodic visualization

## Repository structure
- utils/
  - data_util.py: Materials download, HDF5 I/O, graph building, Dataset/DataLoader, collate with resampling and band masks
  - train_util.py: Params counter and visualization (true vs pred, solid vs dashed), saved as visualization/epoch_{n}_{material_id}.png
  - ema_util.py: Exponential moving average for model parameters
- Models/
  - Predictor.py: PhononPredictor (GNN) producing [B, 120, 256] frequencies
  - loss.py: PhononLoss (masked MSE + smoothness on q-axis differences)
- train.py: Training entry (Accelerate), EMA, periodic logging/visualization, best-model saving
- train.sh: Background training launcher (nohup + accelerate)
- Data/: Generated HDF5 dataset (materials_data.h5) and cache (materials_cache.pkl)
- visualization/: Saved comparison plots per epoch and material
- requirements.txt: Python dependencies

## Reproduction

**Installation**
1) Create a virtual environment and install dependencies:
   - pip install -r requirements.txt
   - For torch-geometric, follow the official installation matrix if wheels are needed for your CUDA/PyTorch version.

2) Materials Project API
   - Get an API key from materialsproject.org
   - You can pass the key explicitly when building the dataset (see below)

**Data preparation**
The dataset builder queries Materials Project for structures and phonon band structures, converts to graphs, and stores to HDF5.

Python snippet:
```python
from utils.data_util import MaterialDataProcessor

api_key = "YOUR_MP_API_KEY"
h5_path = "Data/materials_data.h5"

processor = MaterialDataProcessor(api_key=api_key, disable_progress=False, num_mat=None)  # num_mat to limit size
processor.pipeline(h5_path)  # writes HDF5 under Data/
```

**Data format**
- Graph per material (torch_geometric.data.Data):
  - x: node features [Z, electronegativity, group, row, mass, atomic_radius, vdw_radius]
  - edge_index, edge_attr (Voronoi neighbors + weight), pos, lattice_params
- Frequencies: variable bands and q-points per material
- Collate resamples each material to:
  - frequencies: [120, 256] (zero-padded if fewer bands)
  - band_mask: [120] (1 for valid bands)
  - graph_batch: torch_geometric.data.Batch

**Model**
PhononPredictor (Models/Predictor.py)
- Node embedding -> multiple CrystalGNNLayer (message passing with SiLU + LayerNorm) -> global_mean_pool
- MLP head to predict [120, 256] frequencies per graph
- Default hidden_dim=1024

**Loss**
PhononLoss (Models/loss.py)
- data_loss: masked MSE on [bands, q] grid
- smooth_loss: masked MSE on adjacent q-point differences
- total = 0.8 * data_loss + 0.2 * smooth_loss (configurable)

**Training**
- Uses HuggingFace Accelerate for multi-GPU support
- EMA applied every optimizer step
- Periodic logging and visualization
- Best model saved to Models/best_model.pt

**Quick start**
- Prepare dataset (see “Data preparation” above)
- Start training:
```bash
bash train.sh
# or
accelerate launch train.py
```

**Training behavior**
- Logs every N epochs (log_epoch, default: 10)
- Visualizes every M epochs (vis_epoch, default: 50)
  - Outputs PNGs to visualization/ with filename epoch_{n}_{material_id}.png
  - True curve: solid blue, Predicted: dashed red
- Saves the best checkpoint (lowest running average loss) to Models/best_model.pt

**Reproducibility**
- train.py sets random seeds for torch, numpy, and Python
- Accelerate handles device placement; ensure accelerate config matches your GPUs

## Notes
- The dataset creation step queries the Materials Project; network speed and API quotas will affect runtime
- torch-geometric installation may require specific wheels for your CUDA/PyTorch version
- The default resampling target is [120, 256]; adjust data_util.py if you need different shapes

## Author Information
Author
- Name: Zhao Hongshuo
- Affiliation: Zhejiang University, International Campus (Haining), Class of 2024
- Email: hongshuo.24@intl.zju.edu.cn
- Homepage: https://hasson827.github.io

Advisor
- Name: One-WeeLiat
- Email: weeong@intl.zju.edu.cn
- Homepage: https://zjui.intl.zju.edu.cn/node/781
