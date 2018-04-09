"""
Functions for pre-processing.
"""
import numpy as np
from skimage.color import gray2rgb, rgb2gray
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from common import RESULTS_DIR
from dataset.reader import ScienceDataset


def train_kmeans(split_file, img_folder, n_groups=3):
    dataset = ScienceDataset(split_file, img_folder, mode='test')
    img_ids = dataset.ids
    img_feature = np.zeros((len(dataset), 4))

    # use channel means and whether channels equal or not as 4 features
    for img, idx in dataset:
        # mean for 3 channels
        mean = np.mean(np.mean(img, axis=0), axis=0)
        # channels are equal or not
        if mean[0] == mean[1] == mean[2]:
            is_equal = 1
        else:
            is_equal = 0

        img_feature[idx, :3] = mean/255
        img_feature[idx, 3] = is_equal

    # train a K-means model
    kmeans = KMeans(init='k-means++', n_clusters=n_groups).fit(img_feature)

    # save model
    joblib.dump(kmeans, RESULTS_DIR + '/kmeans.pkl')


def make_img_feature(img):
    img_feature = np.zeros((1, 4))
    # mean for 3 channels
    mean = np.mean(np.mean(img, axis=0), axis=0)
    # channels are equal or not
    if mean[0] == mean[1] == mean[2]:
        is_equal = 1
    else:
        is_equal = 0

    img_feature[:, :3] = mean / 255
    img_feature[:, 3] = is_equal
    return img_feature


def classify_img(model, img_feature):
    img_group = model.predict(img_feature)
    return img_group


def normalize_img(img):
    # normalize image (invert color)
    img_trans = 1 - rgb2gray(img)
    # convert to 3 channels
    img_trans = gray2rgb(img_trans)
    return img_trans

