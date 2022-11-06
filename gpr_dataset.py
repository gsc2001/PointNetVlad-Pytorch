import os
from torch.utils.data import Dataset
from loading_pointclouds import load_pc_file
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
import random

filename = "pointclouds.csv"
pointcloud_fols = "pointclouds/"

class GPRProcessed(Dataset):
    def __init__(self, dataset_folder:str, angle = 0):
        self.df_locations = pd.read_csv(os.path.join(dataset_folder, filename))
        self.dataset_folder = dataset_folder
        self.angle = angle

    
    def __len__(self):
        return self.df_locations.shape[0]
    
    def __getitem__(self, index):
        pcd_file_name = os.path.join(self.dataset_folder, pointcloud_fols, f"{self.df_locations['timestamp'][index]}.bin")
        pcd = load_pc_file(pcd_file_name)

        if self.angle > 0:
            x_angle = random.random() * 2 * self.angle - self.angle
            y_angle = random.random() * 2 * self.angle - self.angle

            r = R.from_euler('xyz', [x_angle,y_angle,0], degrees=True)

            T = np.eye(4)
            T[:3,:3] = r.as_matrix()

            hom_pcd = np.vstack((pcd.T, np.ones((1, pcd.shape[0]))))

            res = T @ hom_pcd

            res /= res[3]

            pcd = res[:3].T

        return {'pcd': pcd, 'northing': self.df_locations['northing'][index], 'easting': self.df_locations['easting'][index]}
