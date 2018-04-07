"""
Pre-process train & test datasets.

"""
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from scipy.misc import imsave
from sklearn.externals import joblib

from dataset.reader import ScienceDataset, multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from common import IMAGE_DIR, RESULTS_DIR, DOWNLOAD_DIR, SPLIT_DIR
from utility.file import read_list_from_file, write_list_to_file
from utility.preprocess_function import make_img_feature, normalize_img, classify_img, train_kmeans


def run_make_test_annotation(split_file, img_folder, out_dir):
    """

    :param split_file: file that records image paths.
    :return:
    """

    ids = read_list_from_file(SPLIT_DIR + split_file, comment='#')

    os.makedirs(out_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in tqdm(range(num_ids)):
        name = ids[i]
        image_file = DOWNLOAD_DIR + '/%s/%s/images/%s.png' % (img_folder, name, name)

        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError

        cv2.imwrite(out_dir + '/images/%s.png' % name, image)


def run_make_train_annotation(split_file, img_folder, out_dir):
    """

    :param split_file: file that records image paths.
    :return:
    """

    ids = read_list_from_file(SPLIT_DIR + split_file, comment='#')

    os.makedirs(out_dir + '/multi_masks', exist_ok=True)
    os.makedirs(out_dir + '/overlays', exist_ok=True)
    os.makedirs(out_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in tqdm(range(num_ids)):
        name = ids[i]
        image_files = glob.glob(DOWNLOAD_DIR + '/%s/%s/images/*.png' % (img_folder, name))
        assert(len(image_files) == 1)
        image_file = image_files[0]

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError

        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_files = glob.glob(DOWNLOAD_DIR + '/%s/%s/masks/*.png' % (img_folder, name))
        mask_files.sort()
        num_masks = len(mask_files)
        for i in range(num_masks):
            mask_file = mask_files[i]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = i+1

        # check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0, 255, 0])
        all = np.hstack((image, contour_overlay, color1_overlay,)).astype(np.uint8)

        np.save(out_dir + '/multi_masks/%s.npy' % name, multi_mask)
        cv2.imwrite(out_dir + '/multi_masks/%s.png' % name, color_overlay)
        cv2.imwrite(out_dir + '/overlays/%s.png' % name, all)
        cv2.imwrite(out_dir + '/images/%s.png' % name, image)


def run_classify_norm_imgs(split_file, img_folder, out_folder, out_split_prefix_list, write_split=True):
    dataset = ScienceDataset(split_file, img_folder=img_folder, mode='test')
    img_ids = dataset.ids

    os.makedirs(IMAGE_DIR + out_folder + '/images', exist_ok=True)

    # use K-means model as classifier
    kmeans = joblib.load(RESULTS_DIR + '/kmeans.pkl')

    img_g1 = []
    img_g2 = []
    img_g3 = []

    for img, idx in tqdm(dataset):
        name = img_ids[idx]
        img_feature = make_img_feature(img)
        img_group = int(classify_img(kmeans, img_feature))

        if img_group == 0:
            img_g1.append(name)
        elif img_group == 1:
            img = normalize_img(img)
            img_g2.append(name)
        else:
            img = normalize_img(img)
            img_g3.append(name)

        imsave(IMAGE_DIR + out_folder + '/images/%s.png' % name, img)

    if write_split:
        write_list_to_file(img_g1, '%s/%s_%d' % (SPLIT_DIR, out_split_prefix_list[0], len(img_g1)))
        write_list_to_file(img_g2, '%s/%s_%d' % (SPLIT_DIR, out_split_prefix_list[1], len(img_g2)))
        write_list_to_file(img_g3, '%s/%s_%d' % (SPLIT_DIR, out_split_prefix_list[2], len(img_g3)))


def make_download_split_file(foler_dir, split_name):

    file_list = sorted(os.listdir(foler_dir))
    write_list_to_file(file_list, '%s/%s_%d' % (SPLIT_DIR, split_name, len(file_list)))


def split_train_val(split_file, val_percentage=0.1):

    img_names = read_list_from_file(SPLIT_DIR + split_file)

    num_val = int(val_percentage * len(img_names))
    val_ids = np.random.choice(len(img_names), num_val, replace=False)

    train_names = []
    val_names = []
    for i, name in enumerate(img_names):
        if i in val_ids:
            val_names.append(name)
        else:
            train_names.append(name)

    write_list_to_file(train_names, SPLIT_DIR + 'train1_train_%d' % len(train_names))
    write_list_to_file(val_names, SPLIT_DIR + 'train1_val_%d' % num_val)


def remove_foler_solit(split_file):
    ids = read_list_from_file(SPLIT_DIR + split_file, comment='#')
    img_list = []
    for idx in ids:
        img_id = idx.split('/')[1]
        img_list.append(img_id)

    write_list_to_file(img_list, SPLIT_DIR + split_file + '_nofolder')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # write train & test split file
#    make_download_split_file(DOWNLOAD_DIR + 'kaggle-dsbowl-2018-dataset-fixes-master/stage1_train', 'train1_fixed_all')
#    make_download_split_file(DOWNLOAD_DIR + 'stage1_test', 'test1_all')

    # run_make_train_annotation(
    #     split_file='train1_fixed_all_664',
    #     img_folder='kaggle-dsbowl-2018-dataset-fixes-master/stage1_train',
    #     out_dir=IMAGE_DIR + 'stage1_train_fixed')

    # run_make_test_annotation(
    #     split_file='test1_all_65',
    #     img_folder='stage1_test',
    #     out_dir=IMAGE_DIR + 'stage1_test')
    #
    # train_kmeans(
    #     split_file='train1_all_670',
    #     img_folder='stage1_train')

    # run_classify_norm_imgs(split_file='train1_fixed_all_664',
    #                        img_folder='stage1_train_fixed',
    #                        out_folder='train1_fixed_norm',
    #                        out_split_prefix_list=[
    #                            'train1_fixed_gray_black',
    #                            'train1_fixed_purple',
    #                            'train1_fixed_gray_white'],
    #                        write_split=True)

    # run_classify_norm_imgs(split_file='test1_all_65',
    #                        img_folder='stage1_test',
    #                        out_folder='test1_norm',
    #                        out_split_prefix_list=[
    #                            'test1_gray_black',
    #                            'test1_purple',
    #                            'test1_gray_white'],
    #                        write_split=True)

    # split_train_val('train1_fixed_all_664')

    remove_foler_solit('train1_ids_gray2_500')
    remove_foler_solit('valid1_ids_gray2_43')

    print('sucess!')
