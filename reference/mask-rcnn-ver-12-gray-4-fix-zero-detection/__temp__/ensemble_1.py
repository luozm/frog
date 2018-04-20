import os, sys
sys.path.append(os.path.dirname(__file__))

from common import *
from dataset.reader import *
from net.layer.box.process import *


def check_valid(proposal, width, height):

    num_proposal = len(proposal)

    keep=[]
    for n in range(num_proposal):
        b,x0,y0,x1,y1,score,label,aux = proposal[n]
        w = x1 - x0
        h = y1 - y0

        valid = 1
        if (w*h <10) or \
           (x0<w/4          and h/w>3.5) or \
           (x1>width-1-w/4  and h/w>3.5) or \
           (y0<h/4          and w/h>3.5) or \
           (y1>height-1-h/4 and w/h>3.5) or \
           0: valid=0

        if valid:
            keep.append(n)

    return keep


#remove border
def make_tight_proposal(proposal, border, width, height):
    proposal = proposal.copy()
    x0 = proposal[:,1]
    y0 = proposal[:,2]
    x1 = proposal[:,3]
    y1 = proposal[:,4]
    w = x1 - x0
    h = y1 - y0
    x = (x1 + x0)/2
    y = (y1 + y0)/2

    aa =  w - h
    bb = (w + h)/(2*border+1)
    ww = np.clip((aa + bb)/2, 0, np.inf)
    hh = np.clip((bb - aa)/2, 0, np.inf)

    is_boundy_x0 = x0 < w/4
    is_boundy_x1 = x1 > width-1 -w/4
    is_boundy_y0 = y0 < h/4
    is_boundy_y1 = y1 > height-1-h/4

    proposal[:, 1] = is_boundy_x0*x0 + (1-is_boundy_x0)*(x - ww/2)
    proposal[:, 2] = is_boundy_y0*y0 + (1-is_boundy_y0)*(y - hh/2)
    proposal[:, 3] = is_boundy_x1*x1 + (1-is_boundy_x1)*(x + ww/2)
    proposal[:, 4] = is_boundy_y1*y1 + (1-is_boundy_y1)*(y + hh/2)
    return proposal

#ensemble =======================================================

class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members=[]
        self.center =[]

    def add_item(self, box, score, instance):
        if self.center ==[]:
            self.members = [{
                'box': box, 'score': score, 'instance': instance
            },]
            self.center  = {
                'box': box, 'score': score, 'union':(instance>0.5), 'inter':(instance>0.5),
            }
        else:
            self.members.append({
                'box': box, 'score': score, 'instance': instance
            })
            center_box   = self.center['box'].copy()
            center_score = self.center['score']
            center_union = self.center['union'].copy()
            center_inter = self.center['inter'].copy()

            self.center['box'] = [
                min(box[0],center_box[0]),
                min(box[1],center_box[1]),
                max(box[2],center_box[2]),
                max(box[3],center_box[3]),
            ]
            self.center['score'] = max(score,center_score)
            self.center['union'] = center_union | (instance>0.5)
            self.center['inter'] = center_inter & (instance>0.5)

    def distance(self, box, score, instance):
        center_box   = self.center['box']
        center_union = self.center['union']
        center_inter = self.center['inter']

        x0 = int(max(box[0],center_box[0]))
        y0 = int(max(box[1],center_box[1]))
        x1 = int(min(box[2],center_box[2]))
        y1 = int(min(box[3],center_box[3]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        if box_intersection<0.01: return 0

        x0 = int(min(box[0],center_box[0]))
        y0 = int(min(box[1],center_box[1]))
        x1 = int(max(box[2],center_box[2]))
        y1 = int(max(box[3],center_box[3]))

        i0 = center_union[y0:y1,x0:x1]  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.logical_and(i0, i1).sum()
        area = np.logical_or(i0, i1).sum()
        overlap = intersection/(area + 1e-12)

        return overlap





def do_clustering( boxes, scores, instances, threshold=0.5):

    clusters = []
    num_arguments   = len(instances)
    for n in range(0,num_arguments):
        box   = boxes[n]
        score = scores[n]
        instance = instances[n]

        num = len(instance)
        for m in range(num):
            b, s, i = box[m],score[m],instance[m]

            is_group = 0
            for c in clusters:
                iou = c.distance(b, s, i)

                if iou>threshold:
                    c.add_item(b, s, i)
                    is_group=1

            if is_group == 0:
                c = Cluster()
                c.add_item(b, s, i)
                clusters.append(c)

    return clusters

#
# def mask_to_more(mask):
#     H,W      = mask.shape[:2]
#     box      = []
#     score    = []
#     instance = []
#
#     for i in range(mask.max()):
#         m = (mask==(i+1))
#
#         #filter by size, boundary, etc ....
#         if 1:
#
#             #box
#             y,x = np.where(m)
#             y0 = y.min()
#             y1 = y.max()
#             x0 = x.min()
#             x1 = x.max()
#             b = [x0,y0,x1,y1]
#
#             #score
#             s = 1
#
#             # add --------------------
#             box.append(b)
#             score.append(s)
#             instance.append(m)
#
#             # image_show('m',m*255)
#             # cv2.waitKey(0)
#
#     box      = np.array(box,np.float32)
#     score    = np.array(score,np.float32)
#     instance = np.array(instance,np.float32)
#
#     if len(box)==0:
#         box      = np.zeros((0,4),np.float32)
#         score    = np.zeros((0,1),np.float32)
#         instance = np.zeros((0,H,W),np.float32)
#
#     return box, score, instance
#
#
# def run_ensemble():
#
#     out_dir = \
#         '/root/share/project/kaggle/science2018/results/__ensemble__/yyy'
#         #'/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/ouput'
#         #'/root/share/project/kaggle/science2018/results/__ensemble__/xxx'
#
#     ensemble_dirs = [
#         #different predictors, test augments, etc ...
#
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/evaluate_test/test1_ids_gray2_53-00011000_model',
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/evaluate_test/test1_ids_gray2_53-00017000_model',
#         # '/root/share/project/kaggle/science2018/results/__submit__/LB-0.523/npys-0.570'
#
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx',
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_horizontal_flip',
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_vertical_flip',
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_scale_1.2',
#         # '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/xxx_scale_0.8',
#
#         # '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/original',
#         # '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/horizontal_flip',
#         # '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/vertical_flip',
#         # '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/scale_1.2',
#         # '/root/share/project/kaggle/science2018/results/__old_3__/ensemble_example/input/scale_0.8',
#
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal',
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal_unsharp',
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal_clahe',
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal_horizontal_flip',
#        #'/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_vertical_flip',
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal_scale_1.5',
#        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-01/predict/yyy_normal_scale_0.5',
#
#     ]
#
#     ## setup  --------------------------
#     os.makedirs(out_dir +'/average_semantic_mask', exist_ok=True)
#     os.makedirs(out_dir +'/cluster_union_mask', exist_ok=True)
#     os.makedirs(out_dir +'/cluster_inter_mask', exist_ok=True)
#     os.makedirs(out_dir +'/ensemble_mask', exist_ok=True)
#     os.makedirs(out_dir +'/ensemble_mask_overlays', exist_ok=True)
#
#
#     names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
#     names = [n.split('/')[-2]for n in names]
#     sorted(names)
#
#     num_ensemble = len(ensemble_dirs)
#     for name in names:
#         #name='1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e'
#         print(name)
#         boxes=[]
#         scores=[]
#         instances=[]
#
#         average_semantic_mask = None
#         for dir in ensemble_dirs:
#             # npy_file = dir +'/%s.npy'%name
#             # mask = np.load(npy_file)
#             png_file   = dir +'/overlays/%s/%s.mask.png'%(name,name)
#             mask_image = cv2.imread(png_file,cv2.IMREAD_COLOR)
#             mask       = image_to_mask(mask_image)
#
#             if average_semantic_mask is None:
#                 average_semantic_mask = (mask>0).astype(np.float32)
#             else:
#                 average_semantic_mask = average_semantic_mask + (mask>0).astype(np.float32)
#
#             # color_overlay = mask_to_color_overlay(mask)
#             # image_show('color_overlay',color_overlay)
#             # image_show('average_semantic_mask',average_semantic_mask*255)
#             # cv2.waitKey(0)
#
#             box, score, instance = mask_to_more(mask)
#             boxes.append(box)
#             scores.append(score)
#             instances.append(instance)
#
#
#         average_semantic_mask = average_semantic_mask/num_ensemble
#         clusters = do_clustering( boxes, scores, instances, threshold=0.3)
#         H,W      = average_semantic_mask.shape[:2]
#
#
#         # <todo> do your ensemble  here! =======================================
#         ensemble_mask = np.zeros((H,W), np.int32)
#         for i,c in enumerate(clusters):
#             num_members = len(c.members)
#             average = np.zeros((H,W), np.float32)  #e.g. use average
#             for n in range(num_members):
#                 average = average + c.members[n]['instance']
#             average = average/num_members
#
#             ensemble_mask[average>0.5] = i+1
#
#         #do some post processing here ---
#         # e.g. fill holes
#         #      remove small fragment
#         #      remove boundary
#         # <todo> do your ensemble  here! =======================================
#
#
#
#
#         # show clustering/ensmeble results
#         cluster_inter_mask = np.zeros((H,W), np.int32)
#         cluster_union_mask = np.zeros((H,W), np.int32)
#         for i,c in enumerate(clusters):
#             cluster_inter_mask[c.center['inter']]=i+1
#             cluster_union_mask[c.center['union']]=i+1
#
#             # image_show('all',all/num_members*255)
#             # cv2.waitKey(0)
#             # pass
#
#         color_overlay0 = mask_to_color_overlay(cluster_inter_mask)
#         color_overlay1 = mask_to_color_overlay(cluster_union_mask)
#         color_overlay2 = mask_to_color_overlay(ensemble_mask)
#         ##-------------------------
#         average_semantic_mask = (average_semantic_mask*255).astype(np.uint8)
#         average_semantic_mask = cv2.cvtColor(average_semantic_mask,cv2.COLOR_GRAY2BGR)
#
#         cv2.imwrite(out_dir +'/average_semantic_mask/%s.png'%(name),average_semantic_mask)
#         cv2.imwrite(out_dir +'/cluster_inter_mask/%s.mask.png'%(name),color_overlay0)
#         cv2.imwrite(out_dir +'/cluster_union_mask/%s.mask.png'%(name),color_overlay1)
#         cv2.imwrite(out_dir +'/ensemble_mask/%s.mask.png'%(name),color_overlay2)
#
#         image_show('average_semantic_mask',average_semantic_mask)
#         image_show('cluster_inter_mask',color_overlay0)
#         image_show('cluster_union_mask',color_overlay1)
#         #image_show('ensemble_mask',color_overlay2)
#
#         if 1:
#             folder = 'stage1_test'
#             image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
#
#             mask = ensemble_mask
#             norm_image      = adjust_gamma(image,2.5)
#             color_overlay   = mask_to_color_overlay(mask)
#             color1_overlay  = mask_to_contour_overlay(mask, color_overlay)
#             contour_overlay = mask_to_contour_overlay(mask, norm_image, [0,255,0])
#             all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
#             image_show('ensemble_mask',all)
#
#             #psd
#             cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s.png'%(name),all)
#             os.makedirs(out_dir +'/ensemble_mask_overlays/%s'%(name), exist_ok=True)
#             cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.png'%(name,name),image)
#             cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.mask.png'%(name,name),color_overlay)
#             cv2.imwrite(out_dir +'/ensemble_mask_overlays/%s/%s.contour.png'%(name,name),contour_overlay)
#
#
#
#         cv2.waitKey(1)
#         # pass


def run_ensemble_box():

    out_dir = \
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05c/predict/xx_ensemble_box'

    ensemble_dirs = [
        'xx57_normal',
        'xx57_flip_transpose_1',
        'xx57_flip_transpose_2',
        'xx57_flip_transpose_3',
        'xx57_flip_transpose_4',
        'xx57_flip_transpose_5',
        'xx57_flip_transpose_6',
        'xx57_flip_transpose_7',

        'xx57_scale_0.8',
        'xx57_scale_1.2',
        'xx57_scale_0.5',
        'xx57_scale_1.8',

        'xx57_scale_0.8_flip_transpose_1',
        'xx57_scale_0.8_flip_transpose_2',
        'xx57_scale_0.8_flip_transpose_3',
        'xx57_scale_0.8_flip_transpose_4',
        'xx57_scale_0.8_flip_transpose_5',
        'xx57_scale_0.8_flip_transpose_6',
        'xx57_scale_0.8_flip_transpose_7',

        'xx57_scale_1.2_flip_transpose_1',
        'xx57_scale_1.2_flip_transpose_2',
        'xx57_scale_1.2_flip_transpose_3',
        'xx57_scale_1.2_flip_transpose_4',
        'xx57_scale_1.2_flip_transpose_5',
        'xx57_scale_1.2_flip_transpose_6',
        'xx57_scale_1.2_flip_transpose_7',
    ]
    ensemble_dirs = [
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05c/predict/%s'%e for e in ensemble_dirs
    ]

    #setup ---------------------------------------
    os.makedirs(out_dir +'/proposal/overlays', exist_ok=True)
    os.makedirs(out_dir +'/proposal/npys', exist_ok=True)


    names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    names = [n.split('/')[-2]for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for name in names:
        print(name)
        image_file = ensemble_dirs[0] +'/overlays/%s/%s.png'%(name,name)
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        height, width= image.shape[:2]

        image1 = image.copy()
        image2 = image.copy()
        image3 = image.copy()

        detections=[]
        tight_detections=[]
        for t,dir in enumerate(ensemble_dirs):
            npy_file = dir +'/detections/%s.npy'%(name)
            detection = np.load(npy_file)
            tight_detection = make_tight_proposal(detection, 0.25, width, height)

            keep = check_valid(tight_detection, width, height)
            #if lenkeep()==0: continue
            detection       = detection[keep]
            tight_detection = tight_detection[keep]

            detections.append(detection)
            tight_detections.append(tight_detection)
            num_detection = len(detection)
            for n in range(num_detection):
                _,x0,y0,x1,y1,score,label,k = tight_detection[n]  #detection[n]

                color = to_color((score-0.5)/0.5,(0,255,255))
                cv2.rectangle(image2, (x0,y0), (x1,y1), color, 1)

                if t == 0:
                    cv2.rectangle(image1, (x0,y0), (x1,y1), color, 1)

            # image_show('image1',image1,1)
            # cv2.waitKey(0)

        detections = np.vstack(detections)
        tight_detections = np.vstack(tight_detections)
        rois = tight_detections[:,1:6]  #detections[:,1:6]
        keep = cython_nms(rois, 0.5)
        # for i in keep:
        #     #_,x0,y0,x1,y1,score,label,k = detections[i]
        #     x0,y0,x1,y1,score = rois[i]
        #     cv2.rectangle(image3, (x0,y0), (x1,y1), (0,255,0), 1)

        nms = detections[keep]
        tight_nms = tight_detections[keep]

        num_nms = len(nms)
        for i in range(num_nms):
            #_,x0,y0,x1,y1,score,label,k = detections[i]
            x0,y0,x1,y1,score = tight_nms[i][1:6]

            #x0,y0,x1,y1 = small_nms_box[n]
            color = to_color((score-0.5)/0.5,(0,255,255))
            cv2.rectangle(image3, (x0,y0), (x1,y1), color, 1)


        draw_shadow_text(image1, 'original',  (5,15),0.5, (255,255,255), 1)
        draw_shadow_text(image2, 'all',  (5,15),0.5, (255,255,255), 1)
        draw_shadow_text(image3, 'all(nms)',  (5,15),0.5, (255,255,255), 1)

        all = np.hstack([image1,image3,image2])

        #image_show('image1',image1,1)
        #image_show('image2',image2,1)
        #image_show('image3',image3,1)
        image_show('all',all,1)
        cv2.waitKey(1)

        ##save results ---------------------------------------
        np.save(out_dir +'/proposal/npys/%s.npy'%(name),nms)
        cv2.imwrite(out_dir +'/proposal/overlays/%s.png'%(name),all)




def run_ensemble_xxx():

    out_dir = \
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05c/predict/xx_ensemble_mask'

    ensemble_dirs = [
        'xx57_mask_only_normal',
        'xx57_mask_only_flip_transpose_1',
        'xx57_mask_only_flip_transpose_2',
        'xx57_mask_only_flip_transpose_3',
        'xx57_mask_only_flip_transpose_4',
        'xx57_mask_only_flip_transpose_5',
        'xx57_mask_only_flip_transpose_6',
        'xx57_mask_only_flip_transpose_7',

        'xx57_mask_only_scale_0.8',
        'xx57_mask_only_scale_1.2',
        'xx57_mask_only_scale_0.5',
        'xx57_mask_only_scale_1.8',

        'xx57_mask_only_scale_0.8_flip_transpose_1',
        'xx57_mask_only_scale_0.8_flip_transpose_2',
        'xx57_mask_only_scale_0.8_flip_transpose_3',
        'xx57_mask_only_scale_0.8_flip_transpose_4',
        'xx57_mask_only_scale_0.8_flip_transpose_5',
        'xx57_mask_only_scale_0.8_flip_transpose_6',
        'xx57_mask_only_scale_0.8_flip_transpose_7',

        'xx57_mask_only_scale_1.2_flip_transpose_1',
        'xx57_mask_only_scale_1.2_flip_transpose_2',
        'xx57_mask_only_scale_1.2_flip_transpose_3',
        'xx57_mask_only_scale_1.2_flip_transpose_4',
        'xx57_mask_only_scale_1.2_flip_transpose_5',
        'xx57_mask_only_scale_1.2_flip_transpose_6',
        'xx57_mask_only_scale_1.2_flip_transpose_7',
    ]
    ensemble_dirs = [
        '/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05c/predict/%s'%e for e in ensemble_dirs
    ]

    #setup ---------------------------------------
    os.makedirs(out_dir +'/proposal/overlays', exist_ok=True)
    os.makedirs(out_dir +'/proposal/npys', exist_ok=True)


    names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    names = [n.split('/')[-2]for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for i,name in enumerate(names):
        print('%05d %s'%(i,name))
        image_file = ensemble_dirs[0] +'/overlays/%s.png'%(name)
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        height, width= image.shape[:2]

        image1 = image.copy()

        overall = np.zeros((height, width,3),np.float32)

        detections=[]
        tight_detections=[]
        for t,dir in enumerate(ensemble_dirs):
            image_file = dir +'/overlays/%s.png'%(name)
            image = cv2.imread(image_file,cv2.IMREAD_COLOR)
            overall += image


        overall = overall/num_ensemble
        cv2.imwrite(out_dir +'/proposal/overlays/%s.png'%(name),overall)

        image_show('overall',overall,1)
        #image_show('image2',image2,1)
        #image_show('image3',image3,1)
        #image_show('all',all,1)
        cv2.waitKey(0)






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_ensemble_box()
    run_ensemble_xxx()
    print('\nsucess!')