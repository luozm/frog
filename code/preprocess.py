"""
Pre-process train & test datasets.

"""
import os
import cv2
import glob
import numpy as np

from dataset.reader import multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from common import DATA_DIR
from utility.file import read_list_from_file


def run_make_test_annotation(split_file, out_dir):
    """

    :param split_file: file that records image paths.
    :return:
    """

    ids = read_list_from_file(DATA_DIR + '/split/' + split_file, comment='#')

    os.makedirs(out_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in range(num_ids):
        folder = ids[i].split('/')[0]
        name = ids[i].split('/')[1]
        image_file = DATA_DIR + '/__download__/%s/%s/images/%s.png' % (folder, name, name)

        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError

        cv2.imwrite(out_dir + '/images/%s.png' % name, image)


def run_make_train_annotation(split_file, out_dir):
    """

    :param split_file: file that records image paths.
    :return:
    """

    ids = read_list_from_file(DATA_DIR + '/split/' + split_file, comment='#')

    os.makedirs(out_dir + '/multi_masks', exist_ok=True)
    os.makedirs(out_dir + '/overlays', exist_ok=True)
    os.makedirs(out_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]

        name = id.split('/')[-1]
        folder = id.split('/')[0]
        image_files = glob.glob(DATA_DIR + '/__download__/%s/%s/images/*.png' % (folder, name))
        assert(len(image_files) == 1)
        image_file = image_files[0]
        print(id)

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError

        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_files = glob.glob(DATA_DIR + '/__download__/%s/%s/masks/*.png' % (folder, name))
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


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

#    run_make_train_annotation(
#        split_file='train1_ids_all_670',
#        out_dir=DATA_DIR + '/image/stage1_train')

    run_make_test_annotation(
        split_file='test1_ids_all_65',
        out_dir=DATA_DIR + '/image/stage1_test')

    print('sucess!')
