# Phonon Predictor

Lightweight PyTorch project to predict phonon band structures from crystal data.

## Quick start

To set up the environment locally and run the code:

1. Clone the repository:
	```bash
	git clone https://github.com/hasson827/phonon_predictor.git
	cd phonon_predictor
	```

2. Create a virtual environment:
	```bash
	conda create -n phonon python=3.9
	conda activate phonon
	```

3. Install the required dependencies:
	```bash
	pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	```
	Replace `${TORCH}` and `${CUDA}` with your specific versions (e.g., `cpu`, `cu118` for CUDA 11.8, and `2.0.0` for PyTorch 2.0). For example:
	```bash
	pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
	conda install conda-forge::mendeleev
	conda install conda-forge::seekpath
	conda install -c conda-forge --yes --force-reinstall pymatgen
	```

4. Train (run in background)

	```bash
	bash train.sh
	```

	Training logs will be written to `train.log`. Model checkpoints are saved to `./models`.

## Files of interest

- `train.py` — main training entrypoint.
- `config_file.py` — basic run configuration (seed, plotting palette, etc.).
- `requirements.txt` — Python dependencies.
- `data/` — raw and processed data (see `data/phonon/`).
- `models/` — model definitions and saved checkpoints.
- `utils/` — helper modules for loading data, training, loss, plotting, etc.

## Reference
**Dataset:** Guido Petretto, Shyam Dwaraknath, Henrique P. C. Miranda, Donald Winston, *et al.* "High-throughput Density-Functional Perturbation Theory phonons for inorganic materials." (2018) figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3938023.v1

## Data Availablility Statement
The data that support the findings of this study are openly available in GitHub at https://github.com/RyotaroOKabe/phonon_prediction. The $\Gamma$-phonon database generated with the MVN method is available at https://osf.io/k5utb/
