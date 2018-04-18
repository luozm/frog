import os
import cv2
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt

from common import SPLIT_DIR, IMAGE_DIR, RESULTS_DIR
from dataset.reader import multi_mask_to_color_overlay
from utility.file import read_list_from_file
from utility.draw import image_show


# predict_ensemble =======================================================

class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members = []
        self.center = {}

    def add_item(self, proposal, instance):
        if self.center == {}:
            self.members = [{
                'proposal': proposal, 'instance': instance
            },]
            self.center = {
                'union_proposal': proposal, 'union_instance':instance,
                'inter_proposal': proposal, 'inter_instance':instance,
            }
        else:
            self.members.append({
                'proposal': proposal, 'instance': instance
            })
            center_union_proposal = self.center['union_proposal'].copy()
            center_union_instance = self.center['union_instance'].copy()
            center_inter_proposal = self.center['inter_proposal'].copy()
            center_inter_instance = self.center['inter_instance'].copy()

            self.center['union_proposal'][1] = max(center_union_proposal[1],proposal[1])
            self.center['union_proposal'][2] = max(center_union_proposal[2],proposal[2])
            self.center['union_proposal'][3] = min(center_union_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_union_proposal[4],proposal[4])
            self.center['union_proposal'][5] = min(center_union_proposal[5],proposal[5])
            self.center['union_instance'] = np.maximum(center_union_instance , instance )

            self.center['inter_proposal'][1] = max(center_inter_proposal[1],proposal[1])
            self.center['inter_proposal'][2] = max(center_inter_proposal[2],proposal[2])
            self.center['inter_proposal'][3] = min(center_inter_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_inter_proposal[4],proposal[4])
            self.center['inter_proposal'][5] = min(center_inter_proposal[5],proposal[5])
            self.center['inter_instance'] = np.minimum(center_inter_instance, instance )

    def compute_iou(self, proposal, instance, type='union'):

        if type=='union':
            center_proposal = self.center['union_proposal']
            center_instance = self.center['union_instance']
        elif type=='inter':
            center_proposal = self.center['inter_proposal']
            center_instance = self.center['inter_instance']
        else:
            raise NotImplementedError

        x0 = int(max(proposal[1],center_proposal[1]))
        y0 = int(max(proposal[2],center_proposal[2]))
        x1 = int(min(proposal[3],center_proposal[3]))
        y1 = int(min(proposal[4],center_proposal[4]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        #if box_intersection<0.01: return 0

        x0 = int(min(proposal[1],center_proposal[1]))
        y0 = int(min(proposal[2],center_proposal[2]))
        x1 = int(max(proposal[3],center_proposal[3]))
        y1 = int(max(proposal[4],center_proposal[4]))

        i0 = center_instance[y0:y1,x0:x1]>0.5  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.minimum(i0, i1).sum()
        area    = np.maximum(i0, i1).sum()
        overlap = intersection/(area + 1e-12)

        if 0: #debug
            m = np.zeros((*instance.shape,3),np.uint8)
            m[:,:,0]=instance*255
            m[:,:,1]=center_instance*255

            cv2.rectangle(m, (x0,y0),(x1,y1),(255,255,255),1)
            image_show('m',m)
            print('%0.5f'%overlap)
            cv2.waitKey(0)

        return overlap


def sort_clsuters(clusters):

    value=[]
    for c in clusters:
        x0,y0,x1,y1 = (c.center['inter_proposal'] + c.center['union_proposal'])[1:5]
        value.append((x0+x1)+(y0+y1)*100000)
    value=np.array(value)
    index = list(np.argsort(value))

    return index


def do_clustering(proposals, instances, threshold=0.3, type='union'):

    clusters = []
    num_augments = len(instances)
    for n in range(0, num_augments):
        proposal = proposals[n]
        instance = instances[n]

        num = len(instance)
        for i in range(num):
            p, m = proposal[i], instance[i]
            # remove small instances
            if m.sum() < 5: continue

            is_found = 0
            max_iou = 0
            for c in clusters:
                iou = c.compute_iou(p, m, type)
                max_iou = max(max_iou, iou)

                if iou > threshold:
                    c.add_item(p, m)
                    is_found = 1
            #print(max_iou)

            if is_found == 0:
                c = Cluster()
                c.add_item(p, m)
                clusters.append(c)
            #print(len(clusters),max_iou)

    return clusters


def remove_small_fragment(mask, min_area):

    N = mask.max()
    for n in range(N):
        m = (mask == n+1)
        label = skimage.morphology.label(m)
        num_labels = label.max()

        if num_labels>0:
            areas = [(label==c+1).sum() for c in range(num_labels)]
            max_area = max(areas)

            for c in range(num_labels):
                if areas[c] != max_area:
                    mask[label==c+1]=0
                # else:
                #     if max_area<min_area:
                #         mask[label==c+1]=0
    return mask


def remove_small_hole(mask):
    mask = skimage.morphology.remove_small_holes(mask)
    return mask


def run_make_l2_data(split_file, out_dir, img_folder, ensemble_folders, threshold=0.7):

    ensemble_dirs = [
        out_dir + 'predict/%s' % e for e in ensemble_folders
    ]

    # setup ---------------------------------------
    os.makedirs(out_dir + 'ensemble/overlays', exist_ok=True)
    os.makedirs(out_dir + 'ensemble/npys', exist_ok=True)
    os.makedirs(out_dir + 'ensemble/multi_masks', exist_ok=True)

    ids = read_list_from_file(SPLIT_DIR + split_file, comment='#')

    num_ensemble = len(ensemble_dirs)
    for i in range(len(ids)):
        name = ids[i]
        print('%05d %s' % (i, name))

        image = cv2.imread(IMAGE_DIR + '%s/images/%s.png' % (img_folder, name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        instances = []
        proposals = []
        for dir in ensemble_dirs:
            instance = np.load(dir + '/instances/%s.npy' % name)
            proposal = np.load(dir + '/detections/%s.npy' % name)
            assert len(proposal) == len(instance)

            instances.append(instance)
            proposals.append(proposal)

        clusters = do_clustering(proposals, instances, type='union', threshold=0.3)

        # remove outliers
        idx = 0
        while idx < len(clusters):
            c = clusters[idx]
            if len(c.members) < num_ensemble/2:
                clusters.pop(idx)
            else:
                idx += 1

        #clusters = clusters[sort_clsuters(clusters)]
        num_clusters = len(clusters)
        print(num_clusters)

        # predict_ensemble instance
        ensemble_instances = []
        ensemble_instance_edges = []
        ensemble_multi_mask = np.zeros((height, width), np.int32)
        for k in range(num_clusters):
            c = clusters[k]
            num_members = len(c.members)

            ensemble_instance = np.zeros((height, width), np.float32)
            ensemble_instance_edge = np.zeros((height, width), np.float32)
            for j in range(num_members):
                m = c.members[j]['instance']

                kernel = np.ones((3, 3), np.float32)
                me = m - cv2.erode(m, kernel)
                md = m - cv2.dilate(m, kernel)
                diff = (me - md)*m

                ensemble_instance += m
                ensemble_instance_edge += diff

            ensemble_instance = ensemble_instance/ensemble_instance.max()
            ensemble_instance[ensemble_instance <= threshold] = 0
            ensemble_instance_edge = ensemble_instance_edge/ensemble_instance_edge.max()
            ensemble_instance_edge[ensemble_instance <= threshold] = 0

            ensemble_multi_mask[ensemble_instance > threshold] = k + 1

            ensemble_instances.append(ensemble_instance)
            ensemble_instance_edges.append(ensemble_instance_edge)

            # debug
            if 1:
                image_show('ensemble_instance', ensemble_instance*255)
#                image_show('ensemble_instance_edge', ensemble_instance_edge*255)
#                 cv2.waitKey(1)
#                 print()
        ensemble_instances = np.array(ensemble_instances)
        ensemble_instance_edges = np.array(ensemble_instance_edges)
        ensemble_multi_mask = remove_small_fragment(ensemble_multi_mask, 1)

        sum_instance = ensemble_instances.sum(axis=0)
        sum_instance_edge = ensemble_instance_edges.sum(axis=0)

        gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray1 = (sum_instance*255).astype(np.uint8)
        gray2 = (sum_instance_edge*255).astype(np.uint8)
        all = np.hstack([gray0, gray1, gray2])
#        image_show('all', all, 1)
#        cv2.waitKey(1)
#        print()

        # save as train data
        np.save(out_dir + 'ensemble/npys/%s.npy' % name, ensemble_multi_mask)
        color_overlay = multi_mask_to_color_overlay(ensemble_multi_mask)
        cv2.imwrite(out_dir + 'ensemble/multi_masks/%s.png' % name, color_overlay)
        cv2.imwrite(out_dir + 'ensemble/overlays/%s.png' % name, all)


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_make_l2_data(
        split_file='valid1_ids_gray2_43_nofolder',
        img_folder='train1_norm',
        out_dir=RESULTS_DIR + 'mask-rcnn-se-resnext50-train500-new/',
        ensemble_folders=[
            'normal',
            'flip_transpose_1',
            'flip_transpose_2',
            'flip_transpose_3',
            'flip_transpose_4',
            'flip_transpose_5',
            'flip_transpose_6',
            'flip_transpose_7',
            'scale_0.5',
            'scale_0.8',
            'scale_1.2',
            'scale_1.8',
        ],
        threshold=0.7
    )
    print('\nsucess!')
