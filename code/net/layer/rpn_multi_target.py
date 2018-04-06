import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable

from utility.draw import image_show
from net.lib.box.overlap.cython_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rpn_multi_nms import rpn_decode, rpn_encode


## debug and draw #############################################################
def normalize(data):
    data = (data-data.min())/(data.max()-data.min())
    return data


def unflat_to_c3(data, num_bases, scales, H, W):
    dtype =data.dtype
    data=data.astype(np.float32)

    datas=[]

    num_scales = len(scales)
    start = 0
    for l in range(num_scales):
        h,w = int(H/scales[l]), int(W/scales[l])
        c = num_bases[l]

        size = h*w*c
        d = data[start:start+size].reshape(h,w,c)
        start = start+size

        if c==1:
            d = d*np.array([1,1,1])

        elif c==3:
            pass

        elif c==4:
            d = np.dstack((
                (d[:,:,0]+d[:,:,1])/2,
                 d[:,:,2],
                 d[:,:,3],
            ))

        elif c==5:
            d = np.dstack((
                 d[:,:,0],
                (d[:,:,1]+d[:,:,2])/2,
                (d[:,:,3]+d[:,:,4])/2,
            ))


        elif c==6:
            d = np.dstack((
                (d[:,:,0]+d[:,:,1])/2,
                (d[:,:,2]+d[:,:,3])/2,
                (d[:,:,4]+d[:,:,5])/2,
            ))

        else:
            raise NotImplementedError

        d = d.astype(dtype)
        datas.append(d)

    return datas


def draw_rpn_target_truth_box(image, truth_box, truth_label):
    image = image.copy()
    for b,l in zip(truth_box, truth_label):
        x0,y0,x1,y1 = np.round(b).astype(np.int32)
        if l >0:
            cv2.rectangle(image,(x0,y0), (x1,y1), (0,0,255), 1)
        else:
            cv2.rectangle(image,(x0,y0), (x1,y1), (255,255,255), 1)
            #draw_dotted_rect(image,(x0,y0), (x1,y1), (0,0,255), )

    return image


def draw_rpn_target_label(cfg, image, window, label, label_assign, label_weight):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    label         = (normalize(label       )*255).astype(np.uint8)
    label_assign  = (normalize(label_assign)*255).astype(np.uint8)
    label_weight  = (normalize(label_weight)*255).astype(np.uint8)
    labels        = unflat_to_c3(label       , num_bases, scales, H, W)
    label_assigns = unflat_to_c3(label_assign, num_bases, scales, H, W)
    label_weights = unflat_to_c3(label_weight, num_bases, scales, H, W)

    all = []
    for l in range(num_scales):
        pyramid = cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l])

        a = np.vstack((
            pyramid,
            labels[l],
            label_assigns[l],
            label_weights[l],
        ))

        a = cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        all.append(a)

    all = np.hstack(all)
    return all


def draw_rpn_target_target(cfg, image, window, target, target_weight):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]

    target_weight = (normalize(target_weight)*255).astype(np.uint8)
    target_weights = unflat_to_c3(target_weight, num_bases, scales, H, W)

    all=[]
    for l in range(num_scales):
        pyramid = cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l])

        a = np.vstack((
            pyramid,
            target_weights[l],
        ))

        a = cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        all.append(a)

    all = np.hstack(all)
    return all


def draw_rpn_target_target1(cfg, image, window, target, target_weight, is_before=False, is_after=True):

    image = image.copy()

    index = np.where(target_weight!=0)[0]
    for i in index:
        w = window[i]
        t = target[i]
        b = rpn_decode(w.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        if is_before:
            cv2.rectangle(image,(w[0], w[1]), (w[2], w[3]), (0,0,255), 1)
            #cv2.circle(image,((w[0]+w[2])//2, (w[1]+w[3])//2),2, (0,0,255), -1, cv2.LINE_AA)

        if is_after:
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0,255,255), 1)

    return image


# target #############################################################

# cpu version
def make_one_rpn_target(cfg, input, window, truth_box, truth_label, debug=False):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    Args:
        cfg: configuration
        input: input image
        window: [num_anchors, (x0, y0, x1, y1)]
        truth_box: [num_gt_boxes, (x0, y0, x1, y1)]
        truth_label: [num_gt_boxes] Integer class IDs.
        debug: debugging mode

    Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """

    num_window = len(window)
    label = np.zeros((num_window,), np.float32)
    label_assign = np.zeros((num_window, ), np.int32)
    label_weight = np.ones((num_window, ), np.float32)
    target = np.zeros((num_window, 4), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)

    # label = np.zeros((cfg.num_train_rpn_anchors,), np.float32)
    # label_weight = np.ones((cfg.num_train_rpn_anchors,), np.float32)
    # # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    # rpn_match = np.zeros((num_window, ), np.int32)

    num_truth_box = len(truth_box)
    if num_truth_box == 0:
        pass
        # label_assign = np.zeros((cfg.num_train_rpn_anchors, ), np.int32)
        # bg_index = np.random.choice(range(num_window), cfg.num_train_rpn_anchors, replace=False)
        # rpn_match[bg_index] = -1
    else:

        _, height, width = input.size()

        # ----------------------------------------------------------------
        # classification for bg & fg
        # ----------------------------------------------------------------

        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in crowd areas.
        overlap = cython_box_overlap(window, truth_box)
        argmax_overlap = np.argmax(overlap, 1)
        max_overlap = overlap[np.arange(num_window), argmax_overlap]

        bg_index = np.where(max_overlap < cfg.rpn_train_bg_thresh_high)[0]
        label[bg_index] = 0
        label_weight[bg_index] = 1

        # 2. Set anchors with high overlap as positive.
        fg_index_high = np.where(max_overlap >= cfg.rpn_train_fg_thresh_low)[0]
        label[fg_index_high] = 1  # <todo> extend to multi-class ... need to modify regression below too
        label_weight[fg_index_high] = 1
        label_assign[...] = argmax_overlap

        # 3. Set an anchor for each GT box (regardless of IoU value).
        # for each truth, window with highest overlap, include multiple maximums
        argmax_overlap = np.argmax(overlap, 0)
        max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]
        argmax_overlap, a = np.where(overlap == max_overlap)

        fg_index_gt = argmax_overlap
        bg_index = np.setdiff1d(bg_index, fg_index_gt)
        label[fg_index_gt] = 1
        label_weight[fg_index_gt] = 1
        label_assign[fg_index_gt] = a

        # don't care------------
        invalid_truth_label = np.where(truth_label < 0)[0]
        invalid_index = np.isin(label_assign, invalid_truth_label) & (label != 0)
        label_weight[invalid_index] = 0
        target_weight[invalid_index] = 0

#         # ----------------------------------------------------------------
#         # random sample fg & bg
#         # ----------------------------------------------------------------
#
# #        bg_index = np.where((label_weight != 0) & (label == 0))[0]
#
        fg_index = np.union1d(fg_index_gt, fg_index_high)
        num_fg = len(fg_index)
        num_bg = len(bg_index)
#
#         # sample fg, no more than 50% fg
#         if num_fg > cfg.num_train_rpn_anchors/2:
#             num_fg = int(cfg.num_train_rpn_anchors/2)
#             fg_index = sorted(np.random.choice(fg_index, num_fg, replace=False))
#
#         rpn_match[fg_index] = 1
#
#         # sample bg
#         num_bg = cfg.num_train_rpn_anchors-num_fg
#         bg_index = sorted(np.random.choice(bg_index, num_bg, replace=False))
#         rpn_match[bg_index] = -1

        # regression ---------------------------------------
        target_window = window[fg_index]
        target_truth_box = truth_box[label_assign[fg_index]]

#        target = rpn_encode(target_window, target_truth_box)
        target[fg_index] = rpn_encode(target_window, target_truth_box)
        target_weight[fg_index] = 1

#        label[:num_fg] = 1
#        target_weight = np.ones((num_fg, ), np.float32)
#        label_assign = label_assign[np.concatenate((fg_index, bg_index))]

        # ----------------------------------------------------------------
        # weight balancing
        # ----------------------------------------------------------------

        # class balancing (use only if don't sample anchors)
        label_weight[fg_index] = 1
        label_weight[bg_index] = num_fg/num_bg
#        label_weight[:num_fg] = 1
#        label_weight[num_fg:] = num_fg/num_bg

        # # <TODO> scale balancing
        # num_scales = len(cfg.rpn_scales)
        # num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]
        # start = 0
        # for l in range(num_scales):
        #     h, w = int(height//2**l), int(width//2**l)
        #     end = start + h*w*num_bases[l]
        #     ## label_weight[start:end] *= (2**l)**2
        #     start = end
        #

        # task balancing
        target_weight[fg_index] = label_weight[fg_index]

        # -----------------------------------
        # debugging
        # -----------------------------------
        if debug:
            image = input.data.cpu().numpy()*255
            image = image.transpose((1, 2, 0)).astype(np.uint8).copy()

            all1 = draw_rpn_target_truth_box(image, truth_box, truth_label)
            all2 = draw_rpn_target_label(cfg, image, window, label, label_assign, label_weight)
            all3 = draw_rpn_target_target(cfg, image, window, target, target_weight)
            all4 = draw_rpn_target_target1(cfg, image, window, target, target_weight)

            image_show('all1', all1, 1)
            image_show('all2', all2, 1)
            image_show('all3', all3, 1)
            image_show('all4', all4, 1)
            cv2.waitKey(0)

    # save
    label = Variable(torch.from_numpy(label)).cuda()
    label_assign = Variable(torch.from_numpy(label_assign)).cuda()
    label_weight = Variable(torch.from_numpy(label_weight)).cuda()
    target = Variable(torch.from_numpy(target)).cuda()
    target_weight = Variable(torch.from_numpy(target_weight)).cuda()
    return label, label_assign, label_weight, target, target_weight

    # if num_truth_box == 0:
    #     target = None
    #     target_weight = None
    # else:
    #     target = Variable(torch.from_numpy(target)).cuda()
    #     target_weight = Variable(torch.from_numpy(target_weight)).cuda()
    # if sum(rpn_match!=0) != cfg.num_train_rpn_anchors:
    #     raise ValueError
    # rpn_match = Variable(torch.from_numpy(rpn_match)).cuda()
    # return label, label_assign, label_weight, target, target_weight, rpn_match


def make_rpn_target(cfg, inputs, window, truth_boxes, truth_labels):

    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []
#    rpn_matchs = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1, -1))
        rpn_label_assigns.append(rpn_label_assign.view(1, -1))
        rpn_label_weights.append(rpn_label_weight.view(1, -1))
        rpn_targets.append(rpn_target.view(1, -1, 4))
        rpn_targets_weights.append(rpn_targets_weight.view(1, -1))
        # if rpn_target is not None:
        #     rpn_targets.append(rpn_target)
        #     rpn_targets_weights.append(rpn_targets_weight)
        # rpn_matchs.append(rpn_match.view(1, -1))

    rpn_labels = torch.cat(rpn_labels, 0)
    rpn_label_assigns = torch.cat(rpn_label_assigns, 0)
    rpn_label_weights = torch.cat(rpn_label_weights, 0)
    rpn_targets = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)
    # rpn_matchs = torch.cat(rpn_matchs, 0)

    return rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights


## check #############################################################
# from dataset.transform import *
# from dataset.reader    import *
#
# def check_layer():
#     image_id = '3ebd2ab34ba86e515feb79ffdeb7fc303a074a98ba39949b905dbde3ff4b7ec0'
#
#     dir = '/root/share/project/kaggle/science2018/data/image/stage1_train'
#     image_file = dir + '/' + image_id + '/images/' + image_id + '.png'
#     npy_file   = dir + '/' + image_id + '/multi_mask.npy'
#
#     multi_mask0 = np.load(npy_file)
#     image0      = cv2.imread(image_file,cv2.IMREAD_COLOR)
#
#     batch_size =4
#     H,W = 256,256
#     images = []
#     multi_masks = []
#     inputs = []
#     boxes  = []
#     labels = []
#     instances = []
#     for b in range(batch_size):
#         image, multi_mask = random_crop_transform2(image0, multi_mask0, W, H)
#         box, label, instance = multi_mask_to_annotation(multi_mask)
#         input = Variable(torch.from_numpy(image.transpose((2,0,1))).float().div(255)).cuda()
#
#         label[[5,12,14,18]]=-1  #dummy ignore
#
#         images.append(image)
#         inputs.append(input)
#         multi_masks.append(multi_mask)
#         boxes.append(box)
#         labels.append(label)
#         instances.append(instance)
#
#         # print information ---
#         N = len(label)
#         for n in range(N):
#             print( '%d  :  %s  %d'%(n, box[n], label[n]),)
#         print('')
#
#     #dummy features
#     in_channels = 256
#     num_heads = 4
#     feature_heights = [ int(H//2**i) for i in range(num_heads) ]
#     feature_widths  = [ int(W//2**i) for i in range(num_heads) ]
#     ps = []
#     for h,w in zip(feature_heights,feature_widths):
#         p = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
#         p = Variable(torch.from_numpy(p)).cuda()
#         ps.append(p)
#
#     #------------------------
#
#     # check layer
#     cfg = type('', (object,), {})() #Configuration() #default configuration
#     cfg.rpn_num_heads  = num_heads
#     cfg.rpn_num_bases  = 3
#     cfg.rpn_base_sizes = [ 8, 16, 32, 64 ] #radius
#     cfg.rpn_base_apsect_ratios = [1, 0.5,  2]
#     cfg.rpn_strides    = [ 1,  2,  4,  8 ]
#
#
#
#     cfg.rpn_train_bg_thresh_high = 0.5
#     cfg.rpn_train_fg_thresh_low  = 0.5
#
#
#     #start here --------------------------
#     bases, windows = make_rpn_windows(cfg, ps)
#     rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights = \
#         make_rpn_target(cfg, inputs, windows, boxes, labels)
#
#


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

    print('sucess')