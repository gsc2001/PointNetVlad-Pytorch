import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    return parser.parse_args()

args = get_args()
runs_folder = os.path.abspath(args.dataset)
filename = "pointclouds.csv"
pointcloud_fols = "pointclouds/"

all_folders = sorted(os.listdir(os.path.join(runs_folder)))

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
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=10)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = (np.setdiff1d(ind_nn[i],list(range(i,i+1))) + offset).tolist()
        negatives = (np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]) + offset).tolist()
        random.shuffle(negatives)
        queries[i + offset] = {"query":query,
                      "positives":positives,"negatives":negatives}
    return queries


# Initialize pandas DataFrame
train_queries = {}
test_queries = {}

cnt_train, cnt_test = 0, 0
for i, folder in enumerate(folders):
    print('Seq:', i, folder)
    df_train = pd.DataFrame(columns=['file','northing','easting'])
    df_test = pd.DataFrame(columns=['file','northing','easting'])

    df_locations = pd.read_csv(os.path.join(runs_folder,folder,filename),sep=',')
    df_locations['timestamp'] = df_locations['timestamp'].apply(lambda ts: os.path.join(runs_folder, folder,pointcloud_fols,str(ts)+'.bin'))
    df_locations = df_locations.rename(columns={'timestamp':'file'})
    print(df_locations['file'])

    for index, row in df_locations.iterrows():
        if index % 3 == 0:
            if index >= int(0.8 * df_locations.shape[0]):
                df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)
    
    train_queries.update(construct_query_dict(df_train, cnt_train))
    test_queries.update(construct_query_dict(df_test,cnt_test))
    cnt_train += df_train.shape[0]
    cnt_test += df_test.shape[0]

    print("Number of training submaps: "+str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

with open('training_queries_baseline.pickle', 'wb') as f:
    pickle.dump(train_queries, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_queries_baseline.pickle', 'wb') as f:
    pickle.dump(test_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
