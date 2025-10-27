import os
import h5py
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch

from mp_api.client import MPRester
from mp_api.client.routes.materials.phonon import PhononBSDOSDoc
PhononBSDOSDoc.model_rebuild()

def safe_float(value, default=0.0) -> float:
    return default if value is None else float(value)

def get_element_features(element) -> List[float]:
    return [
        element.Z,
        safe_float(element.X),  # Electronegativity
        element.group,
        element.row,
        safe_float(element.atomic_mass),
        safe_float(element.atomic_radius),
        safe_float(element.van_der_waals_radius)
    ]

def create_graph_data(structure: Structure) -> Data:
    voronoi_nn = VoronoiNN()
    graph_data = voronoi_nn.get_all_nn_info(structure)
    
    node_features = [get_element_features(site.specie) for site in structure]
    x = torch.tensor(node_features, dtype=torch.float32)
    positions = torch.tensor(np.array([site.coords for site in structure]), dtype=torch.float32)
    lattice_params = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    
    edge_index = []
    edge_attr = []
    
    for i, neighbors in enumerate(graph_data):
        for neighbor in neighbors:
            edge_index.append((i, neighbor['site_index']))
            edge_attr.append(neighbor['weight'])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    
    return Data(
        x=x,
        pos=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lattice_params=lattice_params
    )

def extract_frequencies(phonon_bs) -> torch.Tensor:
    frequencies = torch.tensor(phonon_bs.frequencies, dtype=torch.float32)
    return frequencies

def save_material(group: h5py.Group, material_id: str, graph: Data, frequencies: torch.Tensor) -> None:
    mat_group = group.create_group(material_id)
    graph_group = mat_group.create_group('graph_data')
    for key, tensor in graph.to_dict().items():
        graph_group.create_dataset(key, data=tensor)
    mat_group.create_dataset('frequencies', data=frequencies)

def load_material(group: h5py.Group) -> Dict:
    graph_data = group['graph_data']
    graph_data_dict = {key: torch.tensor(value[()]) for key, value in graph_data.items()}
    graph = Data.from_dict(graph_data_dict)
    frequencies = torch.tensor(group['frequencies'][:])
    return {
        'graph': graph,
        'frequencies': frequencies
    }

def manage_cache(path: str) -> Optional[List]:
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        cache_data = pickle.load(f)
        return cache_data.get('data')


class MaterialDataProcessor:
    def __init__(self, api_key: str, disable_progress: bool, num_mat: Optional[int]=None):
        self.mp = MPRester(api_key, mute_progress_bars=True)
        self.disable_progress = disable_progress
        self.num_mat = num_mat
    
    def find_material(self) -> List[str]:
        docs_phonon = self.mp.materials.phonon.search(fields=["material_id"], chunk_size=1000)
        phonon_ids = [doc.material_id for doc in docs_phonon][:self.num_mat]
        if self.num_mat is not None:
            phonon_ids = phonon_ids[:self.num_mat]
        return phonon_ids
    
    def process_data(self, material_id: str) -> Tuple[str, Optional[Data], Optional[torch.Tensor]]:
        try:
            structure = self.mp.get_structure_by_material_id(material_id)
            phonon_bs = self.mp.get_phonon_bandstructure_by_material_id(material_id)
            graph = create_graph_data(structure)
            frequencies = extract_frequencies(phonon_bs)
            return material_id, graph, frequencies
        except Exception as e:
            print(f"Error processing material {material_id}: {e}")
            return None, None, None
    
    def pipeline(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ids = self.find_material()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(self.process_data, ids), total=len(ids), disable=self.disable_progress))
        with h5py.File(path, 'w') as f:
            mat_group = f.create_group("materials")
            for mat_id, graph, phonon_data in results:
                if graph is not None and phonon_data is not None:
                    save_material(mat_group, mat_id, graph, phonon_data)


class MaterialDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        
        cache_filename = 'materials_cache.pkl'
        cache_path = os.path.join(os.path.dirname(path), cache_filename)
        
        with h5py.File(path, 'r') as f:
            self.material_ids = list(f['materials'].keys())
        
        self.cache_data = manage_cache(cache_path)
        if self.cache_data is None:
            self.cache_data = self.preprocess()
            with open(cache_path, 'wb') as f:
                pickle.dump({'data': self.cache_data}, f)
    
    def __len__(self) -> int:
        return len(self.material_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.cache_data[idx]
    
    def preprocess(self) -> List[Dict]:
        results = []
        for material_id in self.material_ids:
            with h5py.File(self.path, 'r') as f:
                raw_data = load_material(f['materials'][material_id])
                graph = raw_data['graph']
                frequencies = raw_data['frequencies']
                                
                item = {
                    'graph': graph,
                    'frequencies': frequencies,
                    "material_id": material_id
                }
            results.append(item)
        return results


def resample(frequencies: torch.Tensor) -> torch.Tensor:
    actual_bands = min(frequencies.size(0), 120)
    resampled_f_original = F.interpolate(
        frequencies.unsqueeze(0),
        size=256, mode='linear', align_corners=False
    ).squeeze(0)
    resampled_f = torch.zeros(120, 256, dtype=torch.float32)
    resampled_f[:actual_bands] = resampled_f_original[:actual_bands]
    return resampled_f


def collate_fn(batch: List[Dict]):
    graphs = [item['graph'] for item in batch]
    ids = [item['material_id'] for item in batch]
    
    graph_batch = Batch.from_data_list(graphs)
    resampled_fs = []
    band_masks = []
    for item in batch:
        frequencies = item['frequencies']
        resampled_f = resample(frequencies)
        resampled_fs.append(resampled_f)
        
        mask = torch.zeros(120, dtype=torch.float32)
        num_bands = min(frequencies.size(0), 120)
        mask[:num_bands] = 1.0
        band_masks.append(mask)

    resampled_fs = torch.stack(resampled_fs)
    band_masks = torch.stack(band_masks)
        
    return {
        'graph_batch': graph_batch,
        'frequencies': resampled_fs,
        'band_mask': band_masks,
    }, ids

def create_dataloader(dataset: MaterialDataset, batch_size: int):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader