import argparse
import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from gpr_dataset import GPRProcessed
import models.PointNetVlad as PNV
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', required=True)
    parser.add_argument('--model-checkpoint', required=True)
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
    with torch.no_grad():
        for batch in test_loader:
            pcd = batch['pcd']
            pcd = pcd.float()
            pcd = pcd.unsqueeze(1).to(device)
            out = model(pcd)
            out = out.detach().cpu().numpy()
            out = np.squeeze(out)
            print(out.shape)








if __name__ == '__main__':
    main()
