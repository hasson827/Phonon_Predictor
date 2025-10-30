import torch
import pickle as pkl
import numpy as np
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom
import mendeleev as md
from tqdm import tqdm

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
        loaded_dict = pkl.load(file)
    return loaded_dict

def get_lattice_parameters(data):
    a = []
    for i in range(len(data)):
        d = data.iloc[i]
        a.append(d.structure.cell.cellpar()[:3])
    return np.stack(a)

def atom_feature(atomic_number: int, descriptor: str) -> float:
    if descriptor == 'mass':
        return Atom(atomic_number).mass
    elif descriptor == 'number':
        return float(atomic_number)
    else:
        if descriptor == 'radius':
            return md_data.radius[atomic_number]
        elif descriptor == 'en':
            return md_data.pauling[atomic_number]
        elif descriptor == 'ie':
            return md_data.ie[atomic_number]
        elif descriptor == 'dp':
            return md_data.dip[atomic_number]
        else:
            return 1.0

def create_node_input(atomic_numbers: list, descriptor: str) -> torch.Tensor:
    x = []
    for atomic_number in atomic_numbers:
        vec = [0.0] * 118
        vec[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(vec)
    return torch.from_numpy(np.asarray(x, dtype=np.float64))

def get_node_deg(edge_dst, n):
    node_deg = np.zeros((n, 1), dtype=np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += (node_deg == 0)
    return torch.from_numpy(node_deg)

def build_data(mpid: str, structure, real: np.ndarray, r_max: float, qpts: np.ndarray, descriptor: str = 'mass', factor: int = 1000):
    """
    Build data object for graph-based learning models.
    Args:
        mpid (str): Material project ID.
        structure (ase.atoms.Atoms): Atomic structure.
        real (np.ndarray): Real values (e.g., band structure).
        r_max (float): Cutoff radius for neighbor list.
        qpts (np.ndarray): q-points.
        descriptor (str, optional): Descriptor for node features.
        factor (int, optional): Scaling factor for real values.
    """
    symbols = structure.symbols
    positions = torch.from_numpy(structure.positions.copy())
    lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list(
        "ijSDd", a = structure, cutoff = r_max, self_interaction = True
    )
    numb = len(positions)
    
    atomic_numbers = structure.arrays['numbers']
    z = create_node_input(atomic_numbers, 'one-hot')
    x = create_node_input(atomic_numbers, descriptor)
    y = torch.from_numpy(real/factor).unsqueeze(0)
    node_deg = get_node_deg(edge_dst, len(x))
    
    data_dict = {'id': mpid, 'pos': positions, 'lattice': lattice, 'symbol': symbols, 'z': z, 'x': x, 'y': y, 'node_deg': node_deg,
                 'edge_index': torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
                 'edge_shift': torch.tensor(edge_shift, dtype=torch.float64), 'edge_vec': torch.tensor(edge_vec, dtype=torch.float64), 
                 'edge_len': torch.tensor(edge_len, dtype=torch.float64), 'qpts': torch.tensor(qpts, dtype=torch.float64), 
                 'r_max': r_max, 'numb': numb}
    data = Data(**data_dict)
    return data

def generate_data_dict(data, r_max, descriptor: str = 'mass', factor: int = 1000):
    data_dict = dict()
    ids = data['id']
    structures = data['structure']
    qptss = data['qpts']
    reals = data['real_band']
    for id, structure, real, qpts in tqdm(zip(ids, structures, reals, qptss), total=len(ids), desc="Generating data"):
        data_dict[id] = build_data(id, structure, real, r_max, qpts, descriptor, factor)
    return data_dict