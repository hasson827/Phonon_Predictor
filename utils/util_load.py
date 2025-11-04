import glob
import os
import json
import pandas as pd
import numpy as np
from ase import Atoms
from pymatgen.core.structure import Structure

def load_band_structure_data(DIR_CONFIG):
    data_path = os.path.join(DIR_CONFIG['data_dir'], DIR_CONFIG['data_file'])
    if os.path.exists(data_path):
        return pd.read_pickle(data_path)

    rows = []
    for file_path in glob.glob(os.path.join(DIR_CONFIG['raw_dir'], '*.json')):
        with open(file_path) as f:
            data = json.load(f)

        structure = Structure.from_str(data['metadata']['structure'], fmt='cif')
        atoms = Atoms(
            [sp.symbol for sp in structure.species],
            positions=structure.cart_coords.copy(),
            cell=structure.lattice.matrix.copy(),
            pbc=True
        )

        rows.append({
            'id': data['metadata']['material_id'],
            'structure': atoms,
            'qpts': np.array(data['phonon']['qpts']),
            'real_band': np.array(data['phonon']['ph_bandstructure']),
        })

    df = pd.DataFrame(rows)
    df.to_pickle(data_path)
    return df