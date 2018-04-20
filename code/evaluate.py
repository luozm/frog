from common import RESULTS_DIR, PROJECT_PATH, IDENTIFIER, SEED

import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from dataset.transform import pad_to_factor
from dataset.reader import multi_mask_to_annotation, ScienceDataset, instance_to_multi_mask,\
    multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from utility.file import backup_project_as_zip, Logger
from utility.draw import image_show
from net.draw import draw_multi_proposal_metric, draw_mask_metric
from net.metric import compute_precision_for_box, compute_average_precision_for_mask
from net.se_resnext50_mask_rcnn.configuration import Configuration
from net.se_resnext50_mask_rcnn.se_resnext50_mask_rcnn import MaskNet
from dataset.transform import normalize_transform


## overwrite functions ###
def revert(net, images):

    def torch_clip_proposals (proposals, index, width, height):
        boxes = torch.stack((
             proposals[index,0],
             proposals[index,1].clamp(0, width  - 1),
             proposals[index,2].clamp(0, height - 1),
             proposals[index,3].clamp(0, width  - 1),
             proposals[index,4].clamp(0, height - 1),
             proposals[index,5],
             proposals[index,6],
        ), 1)
        return proposals

    # ----

    batch_size = len(images)
    for b in range(batch_size):
        image  = images[b]
        height,width  = image.shape[:2]


        # net.rpn_logits_flat  <todo>
        # net.rpn_deltas_flat  <todo>
        # net.rpn_window       <todo>
        # net.rpn_proposals    <todo>

        # net.rcnn_logits
        # net.rcnn_deltas
        # net.rcnn_proposals <todo>

        # mask --
        # net.mask_logits
        index = (net.detections[:,0]==b).nonzero().view(-1)
        net.detections   = torch_clip_proposals (net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height,:width]

    return net


def eval_augment(image, multi_mask, meta, index):

    pad_image = pad_to_factor(image, factor=16)
    image_norm = normalize_transform(pad_image)
    input = torch.from_numpy(image_norm.transpose((2,0,1))).float()#.div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, image, index


def eval_collate(batch):

    batch_size = len(batch)
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    images    =             [batch[b][5]for b in range(batch_size)]
    indices   =             [batch[b][6]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, images, indices]


# --------------------------------------------------------------
def run_evaluate(val_split, img_folder, mask_folder, out_dir, checkpoint):

    ## setup  ---------------------------
    os.makedirs(out_dir + '/evaluate/overlays', exist_ok=True)
    os.makedirs(out_dir + '/evaluate/npys', exist_ok=True)
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ------------------------------
    cfg = Configuration()
    # cfg.rpn_train_nms_pre_score_threshold = 0.8 #0.885#0.5
    # cfg.rpn_test_nms_pre_score_threshold  = 0.8 #0.885#0.5

    net = MaskNet(cfg).cuda()
    if checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % checkpoint)
        net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
        val_split,
        img_folder=img_folder,
        mask_folder=mask_folder,
        mode='train',
        transform=eval_augment)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_collate)

    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')


    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    mask_average_precisions = []
    mask_precisions_50 = []
    box_precisions_50 = []
    rpn_box_precisions_50 = []

    test_num = 0
#    test_loss = np.zeros(6, np.float32)
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, images, indices) in enumerate(test_loader, 0):
        if all((truth_label > 0).sum() == 0 for truth_label in truth_labels):
            continue

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances)
#            loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)

        ##save results ---------------------------------------
        revert(net, images)

        batch_size = len(indices)
        assert(batch_size == 1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        inputs = inputs.data.cpu().numpy()
        #
        # window          = net.rpn_window
        # rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        # rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        # proposals  = net.rpn_proposals
        masks = net.masks
        rpn_proposals = net.rpn_proposals.cpu().numpy()
        detections = net.detections.cpu().numpy()

        for b in range(batch_size):
            image = images[b]
            height,width = image.shape[:2]
            mask = masks[b]

            rpn_proposal = np.zeros((0, 7), np.float32)
            if len(rpn_proposals) > 0:
                index = np.where(rpn_proposals[:, 0] == b)[0]
                rpn_proposal = rpn_proposals[index]
                rpn_box = rpn_proposal[:, 1:5]

            detection = np.zeros((0, 7), np.float32)
            if len(detections) > 0:
                index = np.where(detections[:, 0] == b)[0]
                detection = detections[index]
                box = detection[:, 1:5]

            truth_mask = instance_to_multi_mask(truth_instances[b])
            truth_box = truth_boxes[b]
            truth_label = truth_labels[b]
            truth_instance = truth_instances[b]

            mask_average_precision, mask_precision =\
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

            rpn_box_precision, _, _, _ = \
                compute_precision_for_box(rpn_box, truth_box, truth_label, threshold=[0.5])
            rpn_box_precision = rpn_box_precision[0]

            box_precision, box_recall, box_result, truth_box_result = \
                compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
            box_precision = box_precision[0]

            mask_average_precisions.append(mask_average_precision)
            mask_precisions_50.append(mask_precision[0][1])
            box_precisions_50.append(box_precision)
            rpn_box_precisions_50.append(rpn_box_precision)

            # --------------------------------------------
            id = test_dataset.ids[indices[b]]
            name =id.split('/')[-1]
#            print('%d\t%s\t%0.5f  (%0.5f)'%(i,name,mask_average_precision, box_precision))
            log.write('%d\t%s\t%0.5f,%0.5f,%0.5f,%0.5f\n'%(i, name, rpn_box_precision, box_precision, mask_precision[0][1], mask_average_precision))

            # ----
            contour_overlay = multi_mask_to_contour_overlay(mask, image, color=[0,255,0])
            color_overlay = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay, color=[255,255,255])
            all1 = np.hstack((image, contour_overlay, color1_overlay))

            all5 = draw_multi_proposal_metric(cfg, image, rpn_proposal, truth_box, truth_label,
                                              [0, 255, 255], [255, 0, 255], [255, 255, 0])
            all6 = draw_multi_proposal_metric(cfg, image, detection, truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
            all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

#            image_show('all1', all1)
#            image_show('all6', all6)
#            image_show('all7', all7)

            cv2.imwrite(out_dir + '/evaluate/%s.rcnn_precision.png' % name, all6)
            cv2.imwrite(out_dir + '/evaluate/%s.mask_precision.png' % name, all7)
            cv2.waitKey(1)

        # print statistics  ------------
        # test_loss += batch_size*np.array((
        #                    loss.cpu().data.numpy(),
        #                    net.rpn_cls_loss.cpu().data.numpy(),
        #                    net.rpn_reg_loss.cpu().data.numpy(),
        #                    net.rcnn_cls_loss.cpu().data.numpy(),
        #                    net.rcnn_reg_loss.cpu().data.numpy(),
        #                    net.mask_cls_loss.cpu().data.numpy(),
        #                  ))
        test_num += batch_size

#    test_loss = test_loss/test_num

    log.write('initial_checkpoint  = %s\n' % checkpoint)
#    log.write('test_loss = %0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % tuple(test_loss))
    log.write('test_num  = %d\n' % test_num)
    log.write('\n')

    mask_average_precisions = np.array(mask_average_precisions)
    mask_precisions_50 = np.array(mask_precisions_50)
    rpn_box_precisions_50 = np.array(rpn_box_precisions_50)
    box_precisions_50 = np.array(box_precisions_50)
    log.write('-------------\n')
    log.write('rpn_box_precision@0.5 = %0.5f\n' % rpn_box_precisions_50.mean())
    log.write('box_precision@0.5 = %0.5f\n' % box_precisions_50.mean())
    log.write('mask_precision@0.5 = %0.5f\n' % mask_precisions_50.mean())
    log.write('mask_average_precision = %0.5f\n' % mask_average_precisions.mean())
    log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    torch.backends.cudnn.benchmark = False
#    run_evaluate('train1_val_67')
    run_evaluate(
        'test1_all_65',
#        'test1_gray_black_53',
#        'valid1_ids_gray2_43_nofolder',
        img_folder='stage1_test',
        mask_folder='stage1_test',
        out_dir=RESULTS_DIR + '/mask-rcnn-se-resnext50-train500-norm-02',
        checkpoint=RESULTS_DIR + '/mask-rcnn-se-resnext50-train500-norm-02/checkpoint/30055_model.pth')

    print('\nsucess!')
