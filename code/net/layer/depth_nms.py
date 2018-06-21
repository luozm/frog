import os
import cv2
import numpy as np

from utility.draw import image_show
from common import np_sigmoid
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
            depth_non_zeros = []
            for i in index:
                area = np.zeros((H, W), np.float32)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1-y0+1, x1-x0+1
                crop = np.array(depth_logits[i], np.float32)
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                area[y0:y1+1, x0:x1+1] = crop

                depth.append(area)
                depth_non_zero = np.zeros((H, W), np.int32)
                depth_non_zero[np.nonzero(area)] += 1
                depth_non_zeros.append(depth_non_zero)

            depth = np.array(depth, np.float32)
            depth_non_zeros = np.array(depth_non_zeros, np.int32)

            depth_sum = depth.sum(axis=0)
            depth_non_zeros = depth_non_zeros.sum(axis=0)
            # for i in range(H):
            #     for j in range(W):
            #         depth_non_zeros[i, j] = sum(depth[:, i, j] != 0)
            depth_mean = depth_sum/(depth_non_zeros + 1e-10)

        depths.append(depth_mean)

    return depths


def depth_revert(cfg, inputs, proposals, mask_logits, depth_logits):

    overlap_threshold = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold = cfg.mask_test_mask_threshold

    proposals = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs = np_sigmoid(mask_logits)

    masks = []
    depths = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]
        mask_temp = []
        depth_temp = []

        if len(index) != 0:

            instance = []
            box = []
            depth = []
            for i in index:
                m = np.zeros((H, W), np.bool)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1 - y0 + 1, x1 - x0 + 1
                label = int(proposals[i, 6])
                crop = mask_probs[i, label]
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                crop = crop > mask_threshold
                m[y0:y1 + 1, x0:x1 + 1] = crop

                instance.append(m)
                box.append((x0, y0, x1, y1))

                area = np.zeros((H, W), np.float32)

                crop = np.array(depth_logits[i], np.float32)
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                area[y0:y1 + 1, x0:x1 + 1] = crop

                depth.append(area)

                # <debug>----------------------------------------------
                if 0:
                    images = inputs.data.cpu().numpy()
                    image = (images[b].transpose((1, 2, 0)) * 255).astype(np.uint8)
                    image = np.clip(image.astype(np.float32) * 4, 0, 255)

                    image_show('image', image, 2)
                    image_show('mask', mask / mask.max() * 255, 2)
                    cv2.waitKey(1)

                    # <debug>----------------------------------------------
            instance = np.array(instance, np.bool)
            box = np.array(box, np.float32)

            # compute overlap
            box_overlap = cython_box_overlap(box, box)

            L = len(index)
            instance_overlap = np.zeros((L, L), np.float32)
            for i in range(L):
                instance_overlap[i, i] = 1
                for j in range(i + 1, L):
                    if box_overlap[i, j] < 0.01: continue

                    x0 = int(min(box[i, 0], box[j, 0]))
                    y0 = int(min(box[i, 1], box[j, 1]))
                    x1 = int(max(box[i, 2], box[j, 2]))
                    y1 = int(max(box[i, 3], box[j, 3]))

                    intersection = (instance[i, y0:y1, x0:x1] & instance[j, y0:y1, x0:x1]).sum()
                    area = (instance[i, y0:y1, x0:x1] | instance[j, y0:y1, x0:x1]).sum()
                    instance_overlap[i, j] = intersection / (area + 1e-12)
                    instance_overlap[j, i] = instance_overlap[i, j]

            # non-max suppress
            score = proposals[index, 5]
            index = list(np.argsort(-score))

            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(index) > 0:
                i = index[0]
                keep.append(i)
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                index = [e for e in index if e not in delete_index]

                # <todo> : merge?

            for i, k in enumerate(keep):
                # mask[np.where(instance[k])] = i + 1
                mask_temp.append(instance[k])
                depth_temp.append(depth[k])

            mask_temp = np.array(mask_temp, np.int32)
            depth_temp = np.array(depth_temp, np.float32)

        if len(depth_temp) == 0:
            depth_temp.append(np.zeros((H, W), np.bool))
            depth_temp = np.array(depth_temp, np.float32)
        depths.append(depth_temp)
        if len(mask_temp) == 0:
            mask_temp.append(np.zeros((H, W), np.bool))
            mask_temp = np.array(mask_temp, np.int32)
        masks.append(mask_temp)
        # masks.append(mask)
    return masks, depths


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



