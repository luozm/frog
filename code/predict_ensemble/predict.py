import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable

from common import DATA_DIR, RESULTS_DIR, PROJECT_PATH, IDENTIFIER, SEED
from net.se_resnext50_mask_rcnn_new.model import MaskRcnnNet
from net.se_resnext50_mask_rcnn_new.configuration import Configuration
from utility.file import Logger, read_list_from_file
from utility.draw import image_show
from dataset.transform_new import do_gamma
from dataset.reader import multi_mask_to_color_overlay
from net.draw import draw_predict_mask, multi_mask_to_contour_overlay
from predict_ensemble.predict_augmentation import do_test_augment_identity, undo_test_augment_identity, \
    do_test_augment_flip_transpose, undo_test_augment_flip_transpose, do_test_augment_scale, undo_test_augment_scale,\
    do_test_augment_scale_flip_transpose, undo_test_augment_scale_flip_transpose


def run_predict(split_file, out_dir, img_folder, initial_checkpoint):

    # augment -----------------------------------------------------------------------------------------------------
    augments = [
        ('normal',           do_test_augment_identity,       undo_test_augment_identity,       {         } ),
        ('flip_transpose_1', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':1,} ),
        ('flip_transpose_2', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':2,} ),
        ('flip_transpose_3', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':3,} ),
        ('flip_transpose_4', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':4,} ),
        ('flip_transpose_5', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':5,} ),
        ('flip_transpose_6', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':6,} ),
        ('flip_transpose_7', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':7,} ),
        # ('scale_0.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.8, 'scale_y': 0.8  } ),
        # ('scale_1.2',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.2, 'scale_y': 1.2  } ),
        # ('scale_0.5',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.5, 'scale_y': 0.5  } ),
        # ('scale_1.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.8, 'scale_y': 1.8  } ),

        # ('scale_0.8_flip_transpose_1',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':1,  } ),
        # ('scale_0.8_flip_transpose_2',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':2,  } ),
        # ('scale_0.8_flip_transpose_3',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':3,  } ),
        # ('scale_0.8_flip_transpose_4',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':4,  } ),
        # ('scale_0.8_flip_transpose_5',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':5,  } ),
        # ('scale_0.8_flip_transpose_6',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':6,  } ),
        # ('scale_0.8_flip_transpose_7',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':7,  } ),
        #
        # ('scale_1.2_flip_transpose_1',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':1,  } ),
        # ('scale_1.2_flip_transpose_2',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':2,  } ),
        # ('scale_1.2_flip_transpose_3',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':3,  } ),
        # ('scale_1.2_flip_transpose_4',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':4,  } ),
        # ('scale_1.2_flip_transpose_5',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':5,  } ),
        # ('scale_1.2_flip_transpose_6',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':6,  } ),
        # ('scale_1.2_flip_transpose_7',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':7,  } ),

    ]

    # -------------------------------------------------------
    # start experiments here!
    # -------------------------------------------------------

    os.makedirs(out_dir + '/backup', exist_ok=True)
#    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ------------------------------
    cfg = Configuration()
    cfg.rcnn_test_nms_pre_score_threshold = 0.5
    cfg.mask_test_nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold

    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n' % (type(net)))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    ids = read_list_from_file(DATA_DIR + '/split/' + split_file, comment='#')
    log.write('\ttsplit   = %s\n' % split_file)
    log.write('\tlen(ids) = %d\n' % len(ids))
    log.write('\n')

    for tag_name, do_test_augment, undo_test_augment, params in augments:

        ## setup  --------------------------
        tag = tag_name
        os.makedirs(out_dir + 'predict/%s/overlays' % tag, exist_ok=True)
        os.makedirs(out_dir + 'predict/%s/predicts' % tag, exist_ok=True)
        os.makedirs(out_dir + 'predict/%s/rcnn_proposals' % tag, exist_ok=True)
        os.makedirs(out_dir + 'predict/%s/detections' % tag, exist_ok=True)
        os.makedirs(out_dir + 'predict/%s/masks' % tag, exist_ok=True)
        os.makedirs(out_dir + 'predict/%s/instances' % tag, exist_ok=True)

        log.write('** start evaluation here @%s! **\n' % tag)
        for i in range(len(ids)):
            name = ids[i]
            print('%03d %s' % (i, name))

            image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png' % (img_folder, name), cv2.IMREAD_COLOR)

            augment_image = do_test_augment(image, proposal=None,  **params)

            net.set_mode('test')
            with torch.no_grad():
                input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                input = Variable(input).cuda()
                net.forward(input)

            rcnn_proposal, detection, mask, instance = undo_test_augment(net, image, **params)

            ##save results ---------------------------------------
            np.save(out_dir + 'predict/%s/rcnn_proposals/%s.npy'%(tag,name),rcnn_proposal)
            np.save(out_dir + 'predict/%s/masks/%s.npy'%(tag,name),mask)
            np.save(out_dir + 'predict/%s/detections/%s.npy'%(tag,name),detection)
            np.save(out_dir + 'predict/%s/instances/%s.npy'%(tag,name),instance)

            if 1:
                norm_image = do_gamma(image, 2.5)

                threshold = 0.8  #cfg.rcnn_test_nms_pre_score_threshold  #0.8
                #all1 = draw_predict_proposal(threshold, image, rcnn_proposal)
                all2 = draw_predict_mask(threshold, image, mask, detection)

                ## save
                #cv2.imwrite(out_dir +'/predict/%s/predicts/%s.png'%(tag,name), all1)
                cv2.imwrite(out_dir + 'predict/%s/predicts/%s.png'%(tag,name), all2)

                #image_show('predict_proposal',all1)
                image_show('predict_mask', all2)

                if 1:
                    norm_image = do_gamma(image, 2.5)
                    color_overlay = multi_mask_to_color_overlay(mask)
                    color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay)
                    contour_overlay = multi_mask_to_contour_overlay(mask, norm_image, [0,255,0])

                    mask_score = instance.sum(0)
                    mask_score = cv2.cvtColor((mask_score/mask_score.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

                    all = np.hstack((image, contour_overlay, color1_overlay, mask_score)).astype(np.uint8)
                    image_show('overlays', all)

                    # psd
                    cv2.imwrite(out_dir + 'predict/%s/overlays/%s.png'%(tag,name),all)

                    os.makedirs(out_dir + 'predict/%s/overlays/%s'%(tag,name), exist_ok=True)
                    cv2.imwrite(out_dir + 'predict/%s/overlays/%s/%s.png'%(tag,name,name),image)
                    cv2.imwrite(out_dir + 'predict/%s/overlays/%s/%s.mask.png'%(tag,name,name),color_overlay)
                    cv2.imwrite(out_dir + 'predict/%s/overlays/%s/%s.contour.png'%(tag,name,name),contour_overlay)

                cv2.waitKey(1)

        log.write('-------------\n')
        log.write('initial_checkpoint  = %s\n' % initial_checkpoint)
        log.write('tag=%s\n' % tag)
        log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    torch.backends.cudnn.benchmark = False

    run_predict(
#        split_file='test1_ids_gray2_53',
        split_file='test1_all_65',
        out_dir=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new/',
        img_folder='test1_norm',
        initial_checkpoint=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new/checkpoint/00049000_model.pth',
    )

    print('\nsucess!')
