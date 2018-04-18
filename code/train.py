"""
Main training script
"""
from common import RESULTS_DIR, SEED, IDENTIFIER, PROJECT_PATH, np_softmax

import os
import cv2
import time
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, ToPILImage
from PIL import Image

from utility.file import Logger, time_to_str
from utility.draw import image_show
from dataset.reader import ScienceDataset, multi_mask_to_annotation, mask_to_inner_contour
from net.learning_rate import get_learning_rate, adjust_learning_rate, StepLR
from net.se_resnext50_mask_rcnn.configuration import Configuration
from net.se_resnext50_mask_rcnn_new.model import MaskRcnnNet
from net.draw import draw_multi_proposal_metric, draw_mask_metric
from dataset.transform import random_shift_scale_rotate_transform2,\
    random_crop_transform2, random_horizontal_flip_transform2,\
    random_vertical_flip_transform2, random_rotate90_transform2, \
    fix_crop_transform2, normalize_transform

from dataset.transform_new import random_transform, random_transform2, do_custom_process1, do_contrast, do_gamma,\
    do_clahe, do_unsharp, do_speckle_noise, do_flip_transpose2, do_elastic_transform2, do_shift_scale_rotate2

# -------------------------------------------------------------------------------------
# common settings
# -------------------------------------------------------------------------------------

# size after resize
WIDTH, HEIGHT = 256, 256


# -------------------------------------------------------------------------------------
# augmentations
# -------------------------------------------------------------------------------------

def train_augment(image, multi_mask, meta, index):
    """train time augmentation

    :param image: image in the dataset
    :param multi_mask: same as image
    :param meta: meta data
    :param index: image index in the dataset
    :return: transformed image & mask
    """

    # illumintaion ------------
    if 1:
        type = random.randint(0, 4)
        if type == 0:
            image = random_transform(image, u=0.5, func=do_custom_process1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])

        elif type == 1:
            image = random_transform(image, u=0.5, func=do_contrast, alpha=[0.5,2.5])

        elif type == 2:
            image = random_transform(image, u=0.5, func=do_gamma, gamma=[1,3])

        elif type == 3:
            image = random_transform(image, u=0.5, func=do_clahe, clip=[1,3], grid=[8,16])

        else:
            pass

    # filter/noise ------------
    if 1:
        type = random.randint(0, 2)
        if type == 0:
            image = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4], alpha=[4,6])

        elif type == 1:
            image = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        else:
            pass

    # geometric ------------
    if 1:
        image, mask = random_transform2(image, multi_mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
#        image, mask = random_transform2(image, mask, u=0.5, func=do_elastic_transform2, grid=[8,64], distort=[0,0.5])

        image, mask = random_crop_transform2(image, mask, WIDTH, HEIGHT, u=0.5)
        image, mask = do_flip_transpose2(image, mask, random.randint(0,8))

    input = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(mask)

    return input, box, label, instance, image, mask, index

    # image, multi_mask = random_shift_scale_rotate_transform2(
    #     image, multi_mask, shift_limit=[0, 0], scale_limit=[1/2, 2],
    #     rotate_limit=[-45, 45], borderMode=cv2.BORDER_REFLECT_101, u=0.5)
    #
    # image, multi_mask = random_horizontal_flip_transform2(image, multi_mask, 0.5)
    # image, multi_mask = random_vertical_flip_transform2(image, multi_mask, 0.5)
    # image, multi_mask = random_rotate90_transform2(image, multi_mask, 0.5)
    # image_crop, multi_mask_crop = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    # image_norm = normalize_transform(image_crop)
    # # if std of image is 0, recrop it
    # while image_norm is None:
    #     image_crop, multi_mask_crop = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    #     image_norm = normalize_transform(image_crop)
    #
    # image_norm = torch.from_numpy(image_norm.transpose((2, 0, 1))).float()#.div(255)
    # box, label, instance = multi_mask_to_annotation(multi_mask_crop)
    #
    # return image_norm, image_crop, box, label, instance, meta, index


def valid_augment(image, multi_mask, meta, index):
    """validation time augmentation

    :param image:
    :param multi_mask:
    :param meta:
    :param index:
    :return:
    """
    # image,  multi_mask = fix_crop_transform2(image, multi_mask, -1, -1, WIDTH, HEIGHT)
    # image_norm = normalize_transform(image)
    # image_norm = torch.from_numpy(image_norm.transpose((2, 0, 1))).float()#.div(255)
    # box, label, instance = multi_mask_to_annotation(multi_mask)
    #
    # return image_norm, image, box, label, instance, multi_mask, index

    image, mask = fix_crop_transform2(image, multi_mask, -1, -1, WIDTH, HEIGHT)
    input = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(mask)

    return input, box, label, instance, image, mask, index


def train_collate(batch):
    """collection function for DataLoader

    :param batch: list of mini-batch samples
    :return: stacked mini-batch samples
    """
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes = [batch[b][1]for b in range(batch_size)]
    labels = [batch[b][2]for b in range(batch_size)]
    instances = [batch[b][3]for b in range(batch_size)]
    images = [batch[b][4] for b in range(batch_size)]
    masks = [batch[b][5]for b in range(batch_size)]
    indices = [batch[b][6]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, images, masks, indices]


def evaluate(net, test_loader):
    """validation function

    :param net:
    :param test_loader:
    :return:
    """

    test_num = 0
    test_loss = np.zeros(6, np.float32)
    test_acc = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, images, truth_masks, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net.forward(inputs, truth_boxes,  truth_labels, truth_instances)
            loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc += 0 #batch_size*acc[0][0]
        test_loss += batch_size*np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
        test_num += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc


def run_train(train_split, val_split, out_dir, resume_checkpoint=None, pretrain_file=None, show_train_img=True):

    # ---------------------------------------------------------------------------
    # setup settings
    # ---------------------------------------------------------------------------

    skip = ['crop', 'mask']

    # make dirs if not exist
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
#    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if resume_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % resume_checkpoint)
        net.load_state_dict(torch.load(resume_checkpoint, map_location=lambda storage, loc: storage))
        #with open(out_dir +'/checkpoint/configuration.pkl','rb') as pickle_file:
        #    cfg = pickle.load(pickle_file)

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file, skip)

    log.write('%s\n\n' % (type(net)))
    log.write('%s\n' % net.version)
    log.write('\n')

    # ---------------------------------------------------------------------------
    # training settings
    # ---------------------------------------------------------------------------

    iter_accum = 1
    batch_size = 4

    num_iters = 100000
    iter_smooth = 20
    iter_log = 50
    iter_valid = 150
    iter_save = [0, num_iters-1]\
                + list(range(0, num_iters, 3000))#1*1000

    # update LR by step
#    LR = None
    LR = StepLR([(0, 0.01),  (55000, 0.001)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch = 0.
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint.replace('_model.pth', '_optimizer.pth'))
        start_iter = checkpoint['iter']
        start_epoch = checkpoint['epoch']

        # load all except learning rate
        rate = get_learning_rate(optimizer)
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)

    # ---------------------------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------------------------

    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
        train_split,
        img_folder='train1_norm',
        mask_folder='stage1_train',
        mode='train',
        transform=train_augment)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    valid_dataset = ScienceDataset(
        val_split,
        img_folder='train1_norm',
        mask_folder='stage1_train',
        mode='train',
        transform=valid_augment)

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n' % (WIDTH, HEIGHT))
    log.write('\ttrain_dataset.split = %s\n' % train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n' % valid_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n' % len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n' % len(valid_dataset))
    log.write('\tlen(train_loader)   = %d\n' % len(train_loader))
    log.write('\tlen(valid_loader)   = %d\n' % len(valid_loader))
    log.write('\tbatch_size  = %d\n' % batch_size)
    log.write('\titer_accum  = %d\n' % iter_accum)
    log.write('\tbatch_size*iter_accum  = %d\n' % batch_size*iter_accum)
    log.write('\n')

    # ---------------------------------------------------------------------------
    # start training
    # ---------------------------------------------------------------------------

    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n' % str(LR))

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss = np.zeros(6, np.float32)
#    train_acc = 0.0
    valid_loss = np.zeros(6, np.float32)
#    valid_acc = 0.0
    batch_loss = np.zeros(6, np.float32)
#    batch_acc = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum_train_acc = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, images, truth_masks, indices in train_loader:
            if all(len(b) == 0 for b in truth_boxes):
                continue

            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            # ---------------------------------------------------------------------------
            # validation
            # ---------------------------------------------------------------------------

            if i % iter_valid == 0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            # ---------------------------------------------------------------------------
            # save model checkpoints
            # ---------------------------------------------------------------------------

            if i in iter_save:
                torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % i)
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % i)
                with open(out_dir + '/checkpoint/configuration.pkl', 'wb') as pickle_file:
                    pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr < 0:
                    break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum

            # one iteration update  -------------
            inputs = Variable(inputs).cuda()
            net.forward(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

            # accumulated update
            loss.backward()
            if j % iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
#            batch_acc = acc[0][0]
            batch_loss = np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
            sum_train_loss += batch_loss
#            sum_train_acc += batch_acc
            sum += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss/sum
#                train_acc = sum_train_acc /sum
                sum_train_loss = np.zeros(6, np.float32)
#                sum_train_acc = 0.
                sum = 0

            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60), i, j, ''), end='', flush=True)#str(inputs.size()))
            j += 1

            # ---------------------------------------------------------------------------
            # print training images
            # ---------------------------------------------------------------------------

            if show_train_img:

                net.set_mode('test')
                with torch.no_grad():
                    net.forward(inputs, truth_boxes, truth_labels, truth_instances)

                batch_size, C, H, W = inputs.size()
#                images = inputs.data.cpu().numpy()
                window = net.rpn_window
                rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
                rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
                rpn_proposals = net.rpn_proposals.data.cpu().numpy()

                rcnn_logits = net.rcnn_logits.data.cpu().numpy()
                rcnn_deltas = net.rcnn_deltas.data.cpu().numpy()
                rcnn_proposals = net.rcnn_proposals.data.cpu().numpy()

                detections = net.detections.data.cpu().numpy()
                masks = net.masks

                #print('train',batch_size)
                for b in range(1):
#                for b in range(batch_size):

                    image = images[b]
                    truth_box = truth_boxes[b]
                    truth_label = truth_labels[b]
                    truth_instance = truth_instances[b]
                    truth_mask = truth_masks[b]

                    rpn_logit_flat = rpn_logits_flat[b]
                    rpn_delta_flat = rpn_deltas_flat[b]
                    rpn_prob_flat = np_softmax(rpn_logit_flat)

                    rpn_proposal = np.zeros((0, 7), np.float32)
                    if len(rpn_proposals) > 0:
                        index = np.where(rpn_proposals[:, 0] == b)[0]
                        rpn_proposal = rpn_proposals[index]

                    rcnn_proposal = np.zeros((0, 7), np.float32)
                    if len(rcnn_proposals) > 0:
                        index = np.where(rcnn_proposals[:, 0] == b)[0]
                        rcnn_logit = rcnn_logits[index]
                        rcnn_delta = rcnn_deltas[index]
                        rcnn_prob = np_softmax(rcnn_logit)
                        rcnn_proposal = rcnn_proposals[index]

                    mask = masks[b]


                    #box = proposal[:,1:5]
                    #mask = masks[b]

                    ## draw --------------------------------------------------------------------------
                    #contour_overlay = multi_mask_to_contour_overlay(truth_mask, image, [255,255,0] )
                    #color_overlay   = multi_mask_to_color_overlay(mask)

                    #all1 = draw_multi_rpn_prob(cfg, image, rpn_prob_flat)
                    #all2 = draw_multi_rpn_delta(cfg, overlay_contour, rpn_prob_flat, rpn_delta_flat, window,[0,0,255])
                    #all3 = draw_multi_rpn_proposal(cfg, image, proposal)
                    #all4 = draw_truth_box(cfg, image, truth_box, truth_label)

                    all5 = draw_multi_proposal_metric(image, rpn_proposal,  truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
                    all6 = draw_multi_proposal_metric(image, rcnn_proposal, truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
                    all7 = draw_mask_metric(image, mask, truth_mask)

                    # image_show('color_overlay',color_overlay,1)
                    # image_show('rpn_prob',all1,1)
                    # image_show('rpn_prob',all1,1)
                    # image_show('rpn_delta',all2,1)
                    # image_show('rpn_proposal',all3,1)
                    # image_show('truth_box',all4,1)
                    # image_show('rpn_precision',all5,1)
                    image_show('rpn_precision', all5, 1)
                    image_show('rcnn_precision', all6, 1)
                    image_show('mask_precision', all7, 1)


                    # summary = np.vstack([
                    #     all5,
                    #     np.hstack([
                    #         all1,
                    #         np.vstack( [all2, np.zeros((H,2*W,3),np.uint8)])
                    #     ])
                    # ])
                    # draw_shadow_text(summary, 'iter=%08d'%i,  (5,3*HEIGHT-15),0.5, (255,255,255), 1)
                    # image_show('summary',summary,1)

                    name = train_dataset.ids[indices[b]].split('/')[-1]
                    #cv2.imwrite(out_dir +'/train/%s.png'%name,summary)
                    #cv2.imwrite(out_dir +'/train/%05d.png'%b,summary)

                    # cv2.imwrite(out_dir + '/train/%05d.rpn_precision.png' % b,  all5)
                    # cv2.imwrite(out_dir + '/train/%05d.rcnn_precision..png' % b, all6)
                    # cv2.imwrite(out_dir + '/train/%05d.mask_precision.png' % b,  all7)
                    cv2.waitKey(1)

                net.set_mode('train')

    # ---------------------------------------------------------------------------
    # save last model
    # ---------------------------------------------------------------------------

    if 1:
        torch.save(net.state_dict(), out_dir + '/checkpoint/%d_model.pth' % i)
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter': i,
            'epoch': epoch,
        }, out_dir + '/checkpoint/%d_optimizer.pth' % i)

    log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train(train_split='train1_ids_gray2_500_nofolder', val_split='valid1_ids_gray2_43_nofolder',
              out_dir=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new',
              pretrain_file=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new/checkpoint/00049000_model.pth',
              show_train_img=True)

    print('\nsucess!')

