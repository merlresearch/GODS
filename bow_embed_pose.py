# Copyright (c) 2019-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

""" Code for embedding poses as a bag-of-words model. The code uses kmeans for learning the words/centroids
 and uses the minimum centroid assignment followed by feature normalization to generate the features
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def filt_data(input):
    output = []
    for item in input:
        if len(item.shape) != 3:
            output.append(np.zeros((1, 25, 3), dtype=np.float32))
        else:
            output.append(item)
    return output


def my_pickle_load(f):
    try:
        return pickle.load(f)
    except:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


parser = argparse.ArgumentParser(description="GODS: pose bow embedding.")
parser.add_argument(
    "--split_num",
    default=0,
    type=int,
    help="in case of cross-validation, specfiy the cross-validation number (1/2/3/4)",
)
parser.add_argument(
    "--gt",
    default="./data/gt/Video_list4.txt",
    type=str,
    help="ground truth file where the pose file name: See extract_pose_from_video argument list",
)
parser.add_argument("--poses_root", default="./", type=str, help="Where are the extracted poses be saved?")
parser.add_argument(
    "--embed_path",
    default="./data/embed/",
    type=str,
    help="path to pkl filename to store the embedded pose data. It will be saved as embed_path/data_train<split_num>.pkl and embed_path/data_test<split_num>.pkl",
)
parser.add_argument("--verbose", action="store_true", help="echo some messages regarding status of the program.")
args = parser.parse_args()

poses_file_list = args.gt  #'Video_list4.txt'
assert int(args.gt[::-1].split(".")[1][0]) == args.split_num

poses_file = open(poses_file_list, "r")
if poses_file == 0:
    print(
        "%s file does not exist! This file is the list of all CD1/normal/20180206143224_000.avi files in this format."
        % (poses_file_list)
    )

# refer to README or https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-c-python
keypt_index_remove = [
    10,
    11,
    13,
    14,
    19,
    20,
    21,
    22,
    23,
    24,
]  # pose keypoint ids to remove from using in the learning setup. We remove lower body parts as they are not visible in the videos.
imsize = [480, 640]  # image size.
num_centroids = 256  # number of centroids.

## pose pre-processing.
if args.verbose:
    print("pose pre-processing!")

pose_data = []
data_va = []
data_tr = []
poses_root = args.poses_root
for video_name in tqdm(poses_file.readlines()):
    video_name = video_name[:-1]
    pose_tem = []

    # normal is class 1 and abnormal is class 0.
    if (
        video_name.split("/")[-2] == "normal"
    ):  # the format of video is assumed to be 'x/normal/video.avi' and x/normal/video.avi/pose.pkl' is where the pose is stored.
        label = 1
    else:
        label = 0

    with open(os.path.join(poses_root, video_name, "pose.pkl"), "rb") as f:
        pose = my_pickle_load(f)
        pose = filt_data(pose)

    if not pose:  # no poses detected for this video.
        continue

    for i in range(len(pose)):
        skeleton = []
        p = pose[i][0]
        skeleton = np.array([item[0:-1] for j, item in enumerate(p) if j not in keypt_index_remove])
        index = np.where(skeleton == 0)

        # normalize and center the poses with skeleton[1] (neck) as the center.
        skeleton[:, 0] = skeleton[:, 0] / imsize[1]
        skeleton[:, 1] = skeleton[:, 1] / imsize[0]
        skeleton[:, 0] = skeleton[:, 0] - skeleton[1][0]
        skeleton[:, 1] = skeleton[:, 1] - skeleton[1][1]
        skeleton[index] = 0

        # if class 'normal':
        if label == 1:
            pose_data.append(skeleton.flatten())
        pose_tem.append(skeleton.flatten())

    if label == 1:
        data_tr.append(pose_tem)  # we use all normal data for learning.
    else:
        data_va.append(pose_tem)

## learning kmeans pose centroids.
data_tr = data_tr
data_va = data_va
if args.verbose:
    print("Running KMeans and learning BoW centroids...")
if not os.path.exists(os.path.join(args.embed_path, "kmeans-centroids.pkl")):
    pose_data = np.array((pose_data))
    kmeans = KMeans(n_clusters=num_centroids).fit(pose_data)
    with open(os.path.join(args.embed_path, "kmeans-centroids.pkl"), "wb") as handle:
        pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(os.path.join(args.embed_path, "kmeans-centroids.pkl"), "rb") as handle:
        kmeans = pickle.load(handle, encoding="latin1")

# pose bow embedding using the learned centroids.
if args.verbose:
    print("Embedding poses using bow model...")
data_train = []
data_test = []

seq_len = 120
stride = 30
beta = 15.0
for pose in data_tr:
    pose = np.array(pose)
    pre = kmeans.predict(pose)
    hist = np.histogram(pre, bins=range(num_centroids + 1))[0]
    hist.astype(float)
    hist = hist / float(pose.shape[0])
    data_train.append(hist)

for pose in data_va:
    pose = np.array(pose)
    pre = kmeans.predict(pose)
    hist = np.histogram(pre, bins=range(num_centroids + 1))[0]
    hist = hist / float(pose.shape[0])

    data_test.append(hist)


# save the embeddings to be used for GODS.
train_embed_file = os.path.join(args.embed_path, "data_train" + str(args.split_num) + ".pkl")
test_embed_file = os.path.join(args.embed_path, "data_test" + str(args.split_num) + ".pkl")
if args.verbose:
    print("saving train and test splits to %s and %s..." % (train_embed_file, test_embed_file))

data_train = np.array(data_train)
data_test = np.array(data_test)
with open(train_embed_file, "wb") as handle:
    pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(test_embed_file, "wb") as handle:
    pickle.dump(data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
