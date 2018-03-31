"""
Classify images first using a naive classifier. And then split datasets into different groups.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from common import DATA_DIR, RESULTS_DIR
from dataset.reader import ScienceDataset
from utility.file import write_list_to_file


def train_classifier(split_file):
    dataset = ScienceDataset(split_file, mode='test')
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
    kmeans = KMeans(init='k-means++', n_clusters=3).fit(img_feature)
    img_group = kmeans.predict(img_feature)

    # split groups
    img_g1 = []
    img_g2 = []
    img_g3 = []
    for idx in range(len(dataset)):
        if img_group[idx] == 0:
            img_g1.append(img_ids[idx])
        elif img_group[idx] == 1:
            img_g2.append(img_ids[idx])
        else:
            img_g3.append(img_ids[idx])

    # write splits to file
    write_list_to_file(img_g1, DATA_DIR + '/split/train1_gray_black_%d' % len(img_g1))
    write_list_to_file(img_g2, DATA_DIR + '/split/train1_purple_%d' % len(img_g2))
    write_list_to_file(img_g3, DATA_DIR + '/split/train1_gray_white_%d' % len(img_g3))

    # save model
    joblib.dump(kmeans, RESULTS_DIR + '/kmeans.pkl')


def classify_imgs(split_file):
    dataset = ScienceDataset(split_file, mode='test')
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

        img_feature[idx, :3] = mean / 255
        img_feature[idx, 3] = is_equal

    # use K-means model as classifier
    kmeans = joblib.load(RESULTS_DIR + '/kmeans.pkl')
    img_group = kmeans.predict(img_feature)

    # split groups
    img_g1 = []
    img_g2 = []
    img_g3 = []
    for idx in range(len(dataset)):
        if img_group[idx] == 0:
            img_g1.append(img_ids[idx])
        elif img_group[idx] == 1:
            img_g2.append(img_ids[idx])
        else:
            img_g3.append(img_ids[idx])

    # write splits to file
    write_list_to_file(img_g1, DATA_DIR + '/split/test1_gray_black_%d' % len(img_g1))
    write_list_to_file(img_g2, DATA_DIR + '/split/test1_purple_%d' % len(img_g2))
    write_list_to_file(img_g3, DATA_DIR + '/split/test1_gray_white_%d' % len(img_g3))


if __name__ == '__main__':
    train_classifier('train1_ids_all_670')
    classify_imgs('test1_ids_all_65')
