import os
import cv2
import numpy as np

from utility.draw import image_show
from net.lib.box.overlap.cython_overlap.cython_box_overlap import cython_box_overlap


def make_empty_depths(inputs):
    """Same as make_empty_masks

    :param inputs:
    :return:
    """
    masks = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks


def depth_merge(cfg, inputs, proposals, depth_logits):
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold

    proposals = proposals.cpu().data.numpy()
    depth_logits = depth_logits.cpu().data.numpy()

    depths = []
    batch_size, _, H, W = inputs.size()
    for b in range(batch_size):
        depth_mean = np.zeros((H, W), np.float32)
        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]

        if len(index) != 0:
            depth = []
            for i in index:
                area = np.zeros((H, W), np.float32)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1-y0+1, x1-x0+1
                crop = depth_logits[i, 0]
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                area[y0:y1+1, x0:x1+1] = crop

                depth.append(area)

            depth = np.array(depth, np.float32)

            depth_sum = depth.sum(axis=0)
            depth_non_zeros = np.zeros((H, W), np.int32)
            for i in range(H):
                for j in range(W):
                    depth_non_zeros[i, j] = sum(depth[:, i, j] != 0)
            depth_mean = depth_sum/depth_non_zeros

        depths.append(depth_mean)

    return depths


def depth_nms(cfg, inputs, proposals, depth_logits):

    overlap_threshold = 0
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold

    proposals = proposals.cpu().data.numpy()
    depth_logits = depth_logits.cpu().data.numpy()

    depths = []
    batch_size, _, H, W = inputs.size()
    for b in range(batch_size):
#        mask = np.zeros((H,W),np.float32)
        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]

        if len(index) != 0:
            depth = []
            box = []
            for i in index:
                area = np.zeros((H, W), np.bool)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1-y0+1, x1-x0+1
                crop = depth_logits[i, 0]
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                area[y0:y1+1, x0:x1+1] = crop

                depth.append(area)
                box.append((x0, y0, x1, y1))

            depth = np.array(depth, np.float32)
            box = np.array(box, np.float32)

            # compute overlap
            box_overlap = cython_box_overlap(box, box)
            #
            # L = len(index)
            # instance_overlap = np.zeros((L,L),np.float32)
            # for i in range(L):
            #     instance_overlap[i,i] = 1
            #     for j in range(i+1,L):
            #         if box_overlap[i,j]<0.01: continue
            #
            #         x0 = int(min(box[i,0],box[j,0]))
            #         y0 = int(min(box[i,1],box[j,1]))
            #         x1 = int(max(box[i,2],box[j,2]))
            #         y1 = int(max(box[i,3],box[j,3]))
            #
            #         intersection = (instance[i,y0:y1,x0:x1] & instance[j,y0:y1,x0:x1]).sum()
            #         area = (instance[i,y0:y1,x0:x1] | instance[j,y0:y1,x0:x1]).sum()
            #         instance_overlap[i,j] = intersection/(area + 1e-12)
            #         instance_overlap[j,i] = instance_overlap[i,j]

            # non-max suppress
            score = proposals[index, 5]
            index = list(np.argsort(-score))

            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(index) > 0:
                i = index[0]
                keep.append(i)
                delete_index = list(np.where(box_overlap[i] > overlap_threshold)[0])
                index = [e for e in index if e not in delete_index]

                #<todo> : merge?

            for i,k in enumerate(keep):
                mask[np.where(instance[k])] = i+1

        masks.append(mask)
    return masks


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))



