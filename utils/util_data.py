import torch
import pickle as pkl
import numpy as np
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom
import mendeleev as md

class MD:
    def __init__(self):
        self.radius, self.pauling, self.ie, self.dip = {}, {}, {}, {}
        for atomic_number in range(1, 119):
            ele = md.element(atomic_number)
            self.radius[atomic_number] = ele.atomic_radius
            self.pauling[atomic_number] = ele.en_pauling
            ie_dict = ele.ionenergies
            self.ie[atomic_number] = ie_dict[min(list(ie_dict.keys()))] if len(ie_dict) > 0 else 0
            self.dip[atomic_number] = ele.dipole_polarizability
md_data = MD()


def pkl_load(filename):
    with open(filename, 'rb') as file:
        return pkl.load(file)


def get_lattice_parameters(data):
    """Get lattice parameters from ASE Atoms objects in a DataFrame."""
    return np.stack([row.structure.cell.cellpar()[:3] for row in data.itertuples(index=False)])


def atom_feature(atomic_number: int, descriptor: str) -> float:
    if descriptor == 'mass':
        return Atom(atomic_number).mass
    if descriptor == 'number':
        return float(atomic_number)
    if descriptor == 'radius':
        return md_data.radius[atomic_number]
    if descriptor == 'en':
        return md_data.pauling[atomic_number]
    if descriptor == 'ie':
        return md_data.ie[atomic_number]
    if descriptor == 'dp':
        return md_data.dip[atomic_number]
    if descriptor == 'one-hot':
        return 1.0
    return float(atomic_number)


def create_node_input(atomic_numbers: list, n=None, descriptor: str = 'mass') -> torch.Tensor:
    N = len(atomic_numbers)
    x = np.zeros((N, 118), dtype=np.float64)
    for i, Z in enumerate(atomic_numbers):
        x[i, int(Z) - 1] = atom_feature(int(Z), descriptor)
    if n is not None:
        temp = np.repeat(x, repeats=N, axis=0)
        x = np.concatenate([x] + [temp], axis=0)
    return torch.from_numpy(x)


def get_node_deg(edge_dst, n):
    deg = np.bincount(edge_dst, minlength=n).astype(np.float64).reshape(-1,1)
    deg += (deg == 0.0)
    return torch.from_numpy(deg)


def gaussian_expansion(edge_len: np.ndarray, K: int = 50,
                       r_max: float = 8.0, sigma: float = 0.5) -> torch.Tensor:
    if edge_len.size == 0:
        return torch.zeros((0, K), dtype=torch.get_default_dtype())
    centers = np.linspace(0.0, float(r_max), int(K), dtype=np.float64)
    diff2 = (edge_len.reshape(-1, 1) - centers.reshape(1, -1)) ** 2
    edge_attr = np.exp(-diff2 / (2.0 * (sigma ** 2)))
    return torch.from_numpy(edge_attr) # [E, K]


def build_data(mpid: str, structure, real: np.ndarray, qpts: np.ndarray, DATA_CONFIG: dict, **kwargs):
    r_max = DATA_CONFIG.get('r_max', 8)
    descriptor = DATA_CONFIG.get('descriptor', 'mass')
    factor = DATA_CONFIG.get('factor', 1000)
    edge_K = DATA_CONFIG.get('edge_K', 50)
    edge_sigma = DATA_CONFIG.get('edge_sigma', 0.5)
    
    symbols = structure.symbols
    positions = torch.from_numpy(structure.positions.copy())
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list(
        "ijSDd", a = structure, cutoff = r_max, self_interaction = True
    )
    edge_attr = gaussian_expansion(edge_len, K=edge_K, r_max=r_max, sigma=edge_sigma)
    edge_vec = edge_vec / np.linalg.norm(edge_vec, axis=-1, keepdims=True).clip(min=1e-9)
    
    numb = len(positions)
    atomic_numbers = structure.arrays['numbers']
    x = create_node_input(atomic_numbers, descriptor=descriptor)
    y = torch.from_numpy(real/factor).unsqueeze(0)
    node_deg = get_node_deg(edge_dst, len(x))
    
    data_dict = {'id': mpid, 
                 'pos': positions, 
                 'lattice': lattice, 
                 'symbol': symbols, 
                 'x': x, 
                 'y': y, 
                 'node_deg': node_deg,
                 'edge_index': torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
                 'edge_vec': torch.tensor(edge_vec, dtype=torch.float64), 
                 'edge_len': edge_attr.clone().detach().to(torch.float64), 
                 'qpts': torch.tensor(qpts, dtype=torch.float64)
    }
    return Data(**data_dict)


def generate_data_dict(data, DATA_CONFIG: dict, **kwargs):
    data_dict = {}
    for id, structure, real, qpts in zip(data['id'], data['structure'], data['real_band'], data['qpts']):
        data_dict[id] = build_data(id, structure, real, qpts, DATA_CONFIG, **kwargs)
    return data_dict