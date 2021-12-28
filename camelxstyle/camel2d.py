import os

import lmdb as lmdb
import numpy as np
from torch.utils.data import Dataset


class Camel2DField(Dataset):
    def __init__(self, data_dir, field_idx, sim_type, set_idx, transforms=[]):
        self.dataset_dir = os.path.join(data_dir, f"Maps_{field_idx}_{sim_type}_{set_idx}_z=0.00")
        self.lmdb_env = lmdb.open(self.dataset_dir, readonly=True, lock=False)
        self.shape = (256, 256)
        self.labels = np.loadtxt(os.path.join(data_dir, f"params_{sim_type}.txt"))
        self.transforms = transforms

    def get_stat(self):
        return self.lmdb_env.stat()

    def get_label(self, idx):
        return self.labels[idx // 15, :].copy()

    def describe(self, idx):
        label = self.get_label(idx)
        print('Value of the parameters for this map')
        print(f'Omega_m: {label[0]:.5f}')
        print(f'sigma_8: {label[1]:.5f}')
        print(f'A_SN1:   {label[2]:.5f}')
        print(f'A_AGN1:  {label[3]:.5f}')
        print(f'A_SN2:   {label[4]:.5f}')
        print(f'A_AGN2:  {label[5]:.5f}')

    def __len__(self):
        return self.get_stat()['entries']

    def __getitem__(self, idx):
        str_idx = f'{idx:08}'
        with self.lmdb_env.begin() as txn:
            data = np.frombuffer(txn.get(str_idx.encode('ascii')), dtype=np.float32).reshape(self.shape).copy()
        for transform in self.transforms:
            data = transform(data)
        label = self.get_label(idx)
        return [data, label]
