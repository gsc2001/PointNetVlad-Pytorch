import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import glob

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--is-test', action='store_true')
    return parser.parse_args()

args = get_args()
runs_folder = os.path.abspath(args.dataset)
filename = "pointclouds.csv"
pointcloud_fols = "pointclouds/"

all_folders = sorted(glob.glob(os.path.join(runs_folder, '*/')))

folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]
p = [p1,p2,p3,p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################



def construct_query_dict(df_centroids, offset):
    tree = KDTree(df_centroids[['northing','easting','z']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting','z']],r=4)
    ind_r = tree.query_radius(df_centroids[['northing','easting','z']], r=10)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = (np.setdiff1d(ind_nn[i],list(range(i-50,i+50))) + offset).tolist()
        positives.sort()
        negatives = (np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]) + offset).tolist()
        negatives.sort()
        queries[i + offset] = {"query":query,
                      "positives":positives,"negatives":negatives}
    return queries


# Initialize pandas DataFrame
train_queries = {}
test_queries = {}

cnt_train, cnt_test = 0, 0
for i, folder in enumerate(folders):
    print('Seq:', i, folder)
    df_train = pd.DataFrame(columns=['file','northing','easting','z'])
    df_test = pd.DataFrame(columns=['file','northing','easting','z'])

    df_locations = pd.read_csv(os.path.join(runs_folder,folder,filename),sep=',')
    df_locations['timestamp'] = df_locations['timestamp'].apply(lambda ts: os.path.join(runs_folder, folder,pointcloud_fols,str(ts)+'.bin'))
    df_locations = df_locations.rename(columns={'timestamp':'file'})

    for index, row in df_locations.iterrows():
        if index % 1 == 0:
            if index >= int(1 * df_locations.shape[0]):
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)
    
    train_queries.update(construct_query_dict(df_train, cnt_train))
    #test_queries.update(construct_query_dict(df_test,cnt_test))
    cnt_train += df_train.shape[0]
    cnt_test += df_test.shape[0]

    print("Number of training submaps: "+str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

_file = 'training_queries_baseline.pickle'
if args.is_test:
    _file = 'test_queries_baseline.pickle'
with open(os.path.join(args.dataset,_file), 'wb') as f:
    pickle.dump(train_queries, f)

