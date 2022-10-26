import argparse
import os
import re
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from gpr_dataset import GPRProcessed
import models.PointNetVlad as PNV
from tqdm import tqdm
from sklearn.neighbors import KDTree


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', required=True)
    parser.add_argument('--model-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=20)
    return parser.parse_args()

def get_model(checkpoint_file:str):

    model = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                      output_dim=256, num_points=4096)
    model = model.to(device)

    print("Evaluating: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    return model

filename = "pointclouds.csv"
pointcloud_fols = "pointclouds/"

def main():
    args = get_args()
    model = get_model(args.model_checkpoint)

    dt = GPRProcessed(args.dataset_folder)

    test_loader = DataLoader(dt,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    # for i_batch in range(int(np.ceil(len(db_idxs) / batch_size))):
        # file_indices = 
    latent_vectors = []    

    with torch.no_grad():
        for batch in test_loader:
            pcd = batch['pcd']
            pcd = pcd.float()
            pcd = pcd.unsqueeze(1).to(device)
            out = model(pcd)
            out = out.detach().cpu().numpy()
            out = np.squeeze(out)
            latent_vectors.append(out)
    
    latent_vectors = np.vstack(latent_vectors)

    base_name = os.path.basename(os.path.normpath(args.dataset_folder))
    print('saving latent_vectors with shape',latent_vectors.shape)
    np.save(f'latent_vectors_{base_name}.npy', latent_vectors)




if __name__ == '__main__':
    main()
