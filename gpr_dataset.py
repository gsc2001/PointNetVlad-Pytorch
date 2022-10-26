import os
from torch.utils.data import Dataset
from loading_pointclouds import load_pc_file
import pandas as pd

filename = "pointclouds.csv"
pointcloud_fols = "pointclouds/"

class GPRProcessed(Dataset):
    def __init__(self, dataset_folder:str):
        self.df_locations = pd.read_csv(os.path.join(dataset_folder, filename))
    
    def __len__(self):
        return self.df_locations.shape[0]
    
    def __getitem__(self, index):
        pcd_file_name = self.df_locations['timestamps'][index]
        pcd = load_pc_file(pcd_file_name)

        return {'pcd': pcd, 'northing': self.df_locations['northing'][index], 'easting': self.df_locations['easting'][index]}
