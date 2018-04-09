from common import *
from net.layer.box.process import*
from utility.draw import *




def make_empty_masks(cfg, mode, inputs):#<todo>
    masks = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks


#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def mask_to_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    return x0, y0, x1, y1

# def mask_nms( cfg, mode, inputs, proposals, mask_logits):
#
#     score_threshold = cfg.mask_test_score_threshold
#     mask_threshold  = cfg.mask_test_mask_threshold
#
#     proposals   = proposals.cpu().data.numpy()
#     mask_logits = mask_logits.cpu().data.numpy()
#     mask_probs  = np_sigmoid(mask_logits)
#
#     masks = []
#     batch_size,C,H,W = inputs.size()
#     for b in range(batch_size):
#         mask  = np.zeros((H,W),np.float32)
#         index = np.where(proposals[:,0]==b)[0]
#
#         instance_id=1
#         if len(index) != 0:
#             for i in index:
#                 p = proposals[i]
#                 prob = p[5]
#                 #print(prob)
#                 if prob>score_threshold:
#                     x0,y0,x1,y1 = p[1:5].astype(np.int32)
#                     h, w = y1-y0+1, x1-x0+1
#                     label = int(p[6]) #<todo>
#                     crop = mask_probs[i, label]
#                     crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
#                     crop = crop>mask_threshold
#
#                     mask[y0:y1+1,x0:x1+1] = crop*instance_id + (1-crop)*mask[y0:y1+1,x0:x1+1]
#                     instance_id = instance_id+1
#
#                 if 0: #<debug>
#
#                     images = inputs.data.cpu().numpy()
#                     image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
#                     image = np.clip(image.astype(np.float32)*4,0,255)
#
#                     image_show('image',image,2)
#                     image_show('mask',mask/mask.max()*255,2)
#                     cv2.waitKey(1)
#
#             #<todo>
#             #non-max-suppression to remove overlapping segmentation
#
#         masks.append(mask)
#     return masks

#
#
# def mask_nms( cfg, mode, inputs, proposals, mask_logits):
#     #images = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)
#
#     overlap_threshold   = cfg.mask_test_nms_overlap_threshold
#     pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
#     mask_threshold      = cfg.mask_test_mask_threshold
#     mask_min_area       = cfg.mask_test_mask_min_area
#
#     proposals   = proposals.cpu().data.numpy()
#     mask_logits = mask_logits.cpu().data.numpy()
#     mask_probs  = np_sigmoid(mask_logits)
#
#     masks = []
#     mask_proposals = []
#     batch_size,C,H,W = inputs.size()
#     for b in range(batch_size):
#         multi_mask = np.zeros((H,W),np.int32)
#         mask_proposal = [ np.zeros((0,8),np.float32),]
#
#         index = np.where((proposals[:,0]==b) & (proposals[:,5]>pre_score_threshold))[0]
#         if len(index) != 0:
#
#             instance=[]
#             box=[]
#             for i in index:
#                 m = np.zeros((H,W),np.bool)
#
#                 x0,y0,x1,y1 = proposals[i,1:5].astype(np.int32)
#                 h, w  = y1-y0+1, x1-x0+1
#                 label = int(proposals[i,6])
#                 crop  = mask_probs[i, label]
#                 crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
#                 crop  = crop > mask_threshold
#
#
#                 # #<debug>
#                 # if 1:
#                 #     image = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)[b].copy()
#                 #     cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
#                 #     image_show('image',image,1)
#                 #     image_show('crop',crop.astype(np.float32)*255,1)
#                 #     cv2.waitKey(0)
#
#
#
#                 # #disallow disconnect components ---------------
#                 # label = skimage.morphology.label(crop)
#                 # num_labels = label.max()
#                 #
#                 # if num_labels!=1:
#                 #     areas = [(label==c+1).sum() for c in range(num_labels)]
#                 #     max_area = max(areas)
#                 #
#                 #     for c in range(num_labels):
#                 #         if areas[c] != max_area:
#                 #             crop[label==c+1]=0
#                 #
#                 # #fill hole
#                 # #  https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
#                 # floodfill = crop.copy().astype(np.uint8)
#                 # h, w = floodfill.shape[:2]
#                 # cv2.floodFill(floodfill,  np.zeros((h+2, w+2), np.uint8), (0,0), 255);
#                 # floodfill_inv = cv2.bitwise_not(floodfill)
#                 # crop = crop | (floodfill_inv>0)
#
#
#
#                 #-----------------------------------------------
#                 #<debug>
#                 if 0:
#                     image = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)[b].copy()
#                     cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
#                     image_show('image',image,1)
#                     image_show('crop',crop.astype(np.float32)*255,1)
#                     cv2.waitKey(0)
#
#
#                 m[y0:y1+1,x0:x1+1] = crop
#
#                 #print(area)
#                 instance.append(m)
#                 box.append((x0,y0,x1,y1))
#
#             instance = np.array(instance,np.bool)
#             box      = np.array(box, np.float32)
#
#             #compute overlap
#             box_overlap = cython_box_overlap(box, box)
#
#             L = len(index)
#             instance_overlap = np.zeros((L,L),np.float32)
#             for i in range(L):
#                 instance_overlap[i,i] = 1
#                 for j in range(i+1,L):
#                     if box_overlap[i,j]<0.01: continue
#
#                     x0 = int(min(box[i,0],box[j,0]))
#                     y0 = int(min(box[i,1],box[j,1]))
#                     x1 = int(max(box[i,2],box[j,2]))
#                     y1 = int(max(box[i,3],box[j,3]))
#
#                     intersection = (instance[i,y0:y1,x0:x1] & instance[j,y0:y1,x0:x1]).sum()
#                     area = (instance[i,y0:y1,x0:x1] | instance[j,y0:y1,x0:x1]).sum()
#                     instance_overlap[i,j] = intersection/(area + 1e-12)
#                     instance_overlap[j,i] = instance_overlap[i,j]
#
#             #non-max suppress
#             score = proposals[index,5]
#             sort  = list(np.argsort(-score))
#
#             ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
#             keep = []
#             while len(sort) > 0:
#                 i = sort[0]
#                 keep.append(i)
#                 delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
#                 sort =  [e for e in sort if e not in delete_index]
#
#                 #<todo> : merge?
#
#             for i,k in enumerate(keep):
#                 multi_mask[np.where(instance[k])] = i+1
#                 b,x0,y0,x1,y1,score,label,_ = proposals[index][k]
#                 mask_proposal.append(np.array([b,x0,y0,x1,y1,score,label,k],np.float32))
#
#         ##remove small fragments -------------------- <todo> this is in complete code
#         if 1:
#             N = multi_mask.max()
#             for n in range(N):
#                 m  = multi_mask==n+1
#                 label = skimage.morphology.label(m)
#                 num_labels = label.max()
#
#                 if num_labels>0:
#                     areas = [(label==c+1).sum() for c in range(num_labels)]
#                     max_area = max(areas)
#
#                     for c in range(num_labels):
#                         if areas[c] != max_area:
#                             multi_mask[label==c+1]=0
#                         else:
#                             if max_area<mask_min_area:
#                                 multi_mask[label==c+1]=0
#
#         ##remove small fragments --------------------
#
#
#         mask_proposal = np.vstack(mask_proposal)
#         mask_proposals.append(mask_proposal)
#         masks.append(multi_mask)
#
#         #print(multi_mask.max(),len(mask_proposal), keep)
#         #assert(multi_mask.max()==len(mask_proposal))
#
#
#     mask_proposals = Variable(torch.from_numpy(np.vstack(mask_proposals))).cuda()
#     return masks, mask_proposals

def remove_small_fragment(mask, min_area):

    N = mask.max()
    for n in range(N):
        m  = mask==n+1
        label = skimage.morphology.label(m)
        num_labels = label.max()

        if num_labels>0:
            areas = [(label==c+1).sum() for c in range(num_labels)]
            max_area = max(areas)

            for c in range(num_labels):
                if areas[c] != max_area:
                    mask[label==c+1]=0
                else:
                    if max_area<min_area:
                        mask[label==c+1]=0
    return mask



def mask_nms( cfg, mode, inputs, proposals, mask_logits):
    assert(len(proposals)==len(mask_logits))

    overlap_threshold   = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold      = cfg.mask_test_mask_threshold
    mask_min_area       = cfg.mask_test_mask_min_area

    proposals   = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs  = np_sigmoid(mask_logits)

    masks = []
    mask_proposals = []
    batch_size,C,H,W = inputs.size()

    for b in range(batch_size):
        mask = np.zeros((H,W),np.int32)
        mask_proposal = [ np.zeros((0,8),np.float32),]


        index = np.where((proposals[:,0]==b) & (proposals[:,5]>pre_score_threshold))[0]
        if len(index) != 0:
            instance = []
            instance_score = []
            box      = []
            for i in index:
                m = np.zeros((H,W),np.bool)
                s = np.zeros((H,W),np.float32)

                x0,y0,x1,y1 = proposals[i,1:5].astype(np.int32)
                h, w  = y1-y0+1, x1-x0+1
                label = int(proposals[i,6])
                crop  = mask_probs[i, label]
                crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)


                #-----------------------------------------------
                if 0: #<debug>
                    image = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)[b].copy()
                    cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
                    image_show('image',image,1)
                    image_show('crop',crop.astype(np.float32)*255,1)
                    cv2.waitKey(0)
                #-----------------------------------------------

                m[y0:y1+1,x0:x1+1] = crop > mask_threshold
                s[y0:y1+1,x0:x1+1] = crop

                instance.append(m)
                box.append([x0,y0,x1,y1])

            # compute overlap ## ================================
            L = len(index)

            box              = np.array(box, np.float32)
            box_overlap      = cython_box_overlap(box, box)
            instance_overlap = np.zeros((L,L),np.float32)
            for i in range(L):
                instance_overlap[i,i] = 1
                for j in range(i+1,L):
                    if box_overlap[i,j]<0.01: continue

                    x0 = int(min(box[i,0],box[j,0]))
                    y0 = int(min(box[i,1],box[j,1]))
                    x1 = int(max(box[i,2],box[j,2]))
                    y1 = int(max(box[i,3],box[j,3]))

                    mi = instance[i][y0:y1,x0:x1]
                    mj = instance[j][y0:y1,x0:x1]

                    intersection = (mi & mj).sum()
                    area = (mi | mj).sum()
                    instance_overlap[i,j] = intersection/(area + 1e-12)
                    instance_overlap[j,i] = instance_overlap[i,j]

            #non-max suppress
            score = proposals[index,5]
            sort  = list(np.argsort(-score))

            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(sort) > 0:
                i = sort[0]
                keep.append(i)
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                sort =  [e for e in sort if e not in delete_index]

                #<todo> : merge?


            for i,k in enumerate(keep):
                t = index[k]
                j = np.where(instance[k])
                mask[j] = i+1

                b,x0,y0,x1,y1,score,label,_ = proposals[t]
                mask_proposal.append(np.array([b,x0,y0,x1,y1,score,label,t],np.float32))


        ## post process -------------------- <todo> this is in complete code
        if 1:
            mask = remove_small_fragment(mask, mask_min_area)
            #mask = fill_hole(mask)

        ##remove small fragments --------------------


        mask_proposal = np.vstack(mask_proposal)
        mask_proposals.append(mask_proposal)
        masks.append(mask)

        #print(multi_mask.max(),len(mask_proposal), keep)
        #assert(multi_mask.max()==len(mask_proposal))

    mask_proposals = Variable(torch.from_numpy(np.vstack(mask_proposals))).cuda()
    return masks, mask_proposals

##-----------------------------------------------------------------------------  
#if __name__ == '__main__':
#    print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#
# 
 
