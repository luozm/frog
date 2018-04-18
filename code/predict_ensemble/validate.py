from common import SPLIT_DIR, IMAGE_DIR, RESULTS_DIR, PROJECT_PATH, IDENTIFIER, SEED

import os
import cv2
import numpy as np

from dataset.reader import multi_mask_to_annotation, ScienceDataset, instance_to_multi_mask,\
    multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from utility.file import Logger, read_list_from_file
from utility.draw import image_show
from net.draw import draw_multi_proposal_metric, draw_mask_metric
from net.metric import compute_precision_for_box, compute_average_precision_for_mask


# --------------------------------------------------------------
def run_evaluate(split_file, truth_mask_folder, mask_folder, out_dir):

    log = Logger()
    log.open(out_dir+'log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    ids = read_list_from_file(SPLIT_DIR + split_file, comment='#')

    log.write('\ttest_dataset.split = %s\n' % split_file)
    log.write('\tlen(test_dataset)  = %d\n' % len(ids))
    log.write('\n')

    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    mask_average_precisions = []

    for i, name in enumerate(ids):
        truth_mask = np.load(IMAGE_DIR + '/%s/multi_masks/%s.npy' % (truth_mask_folder, name)).astype(np.int32)

        mask = np.load(out_dir + mask_folder + '%s.npy' % name).astype(np.int32)

        mask_average_precision, mask_precision =\
            compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

        mask_average_precisions.append(mask_average_precision)

        log.write('%d\t%s\t%0.5f\n'%(i, name, mask_average_precision))

    mask_average_precisions = np.array(mask_average_precisions)
    log.write('-------------\n')
    log.write('mask_average_precision = %0.5f\n' % mask_average_precisions.mean())
    log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_evaluate(
    #     split_file='valid1_ids_gray2_43_nofolder',
    #     truth_mask_folder='stage1_train',
    #     mask_folder='predict/normal/masks/',
    #     out_dir=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new/',
    # )

    run_evaluate(
        split_file='test1_all_65',
        truth_mask_folder='stage1_test',
        mask_folder='submit/npys/',
        out_dir=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-norm-01/',
    )

    print('\nsucess!')
