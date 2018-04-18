from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *

from net.layer.box.process import *

#data reader  ----------------------------------------------------------------
class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()

        self.split     = split
        self.transform = transform
        self.mode      = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')
        self.ids = ids

        #print
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id = self.ids[index]
        folder, name   = id.split('/')
        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            mask = np.load( DATA_DIR + '/image/%s/masks/%s.npy'%(folder,name)).astype(np.int32)
            if self.transform is not None:
                return self.transform(image, mask, index)
            else:
                return image, mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)



# draw  ----------------------------------------------------------------

def color_overlay_to_mask(image):
    H,W = image.shape[:2]

    mask = np.zeros((H,W),np.int32)
    unique_color = set( tuple(v) for m in image for v in m )

    #print(len(unique_color))
    count=0
    for color in unique_color:
        #print(color)
        if color ==(0,0,0): continue

        thresh = (image==color).all(axis=2)
        label  = skimage.morphology.label(thresh)

        index = [label!=0]
        count = mask.max()
        mask[index] =  label[index]+count

    return mask


def mask_to_color_overlay(mask, image=None, color=None):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_instances))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_instances) ]

    for i in range(num_instances):
        overlay[mask==i+1]=color[i]

    return overlay



def mask_to_contour_overlay(mask, image=None, color=[255,255,255]):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    for i in range(num_instances):
        overlay[mask_to_inner_contour(mask==i+1)]=color

    return overlay

# modifier  ----------------------------------------------------------------

def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


##-------------------------------------------------------------------------
MIN_AREA = 5
MIN_SIZE = 5
BORDER   = 0.25
MAX_SIZE = np.inf


def mask_to_annotation(mask, min_area=MIN_AREA, border=BORDER, min_size=MIN_SIZE, max_size=MAX_SIZE ):
    H,W      = mask.shape[:2]
    box      = []
    label    = []
    instance = []

    for i in range(mask.max()):
        m = (mask==(i+1))
        if m.sum()>min_area:

            y,x = np.where(m)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1

            b = round(border*(w+h)/2)
            #border = 0

            x0 = max(0,  x0-b)  #clip
            y0 = max(0,  y0-b)
            x1 = min(W-1,x1+b)
            y1 = min(H-1,y1+b)

            #label
            l = 1 #<todo> support multiclass later ... ?

            if is_small_box((x0,y0,x1,y1),min_size):
                continue
            elif is_big_box((x0,y0,x1,y1),max_size):
                continue

            # add --------------------
            box.append([x0,y0,x1,y1])
            label.append(l)
            instance.append(m)

    box      = np.array(box,np.float32)
    label    = np.array(label,np.float32)
    instance = np.array(instance,np.float32)
    if len(box)==0:
        box      = np.zeros((0,4),np.float32)
        label    = np.zeros((0,1),np.float32)
        instance = np.zeros((0,H,W),np.float32)

    return box, label, instance


def instance_to_mask(instance):
    H,W = instance.shape[1:3]
    mask = np.zeros((H,W),np.int32)

    num_instances = len(instance)
    for i in range(num_instances):
         mask[instance[i]>0] = i+1

    return mask

def mask_to_instance(mask):
    H,W = mask.shape[:2]
    mask = np.zeros((H,W),np.int32)

    num_instances = mask.max()
    instance = np.zeros((num_instances,H,W), np.float32)
    for i in range(num_instances):
         instance[i] = mask==i+1

    return instance






# check ##################################################################################3

def run_check_train_dataset_reader():

    dataset = ScienceDataset(
        'train1_ids_gray2_500',
        #'disk0_ids_dummy_9',
        #'merge1_1',
        mode='train',transform = None,
    )

    for n in range(len(dataset)):
        i=13  #=n
        image, truth_mask, index = dataset[i]

        folder, name   = dataset.ids[index].split( '/')
        print('%05d %s' %(i,name))

        # image1 = random_transform(image, u=0.5, func=process_gamma, gamma=[0.8,2.5])
        # image2 = process_gamma(image, gamma=2.5)

        #image1 = random_transform(image, u=0.5, func=do_process_custom1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])
        #image1 = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4],alpha=[4,6])
        #image1 = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        #image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_elastic_transform2, grid=[16,64], distort=[0,0.5])



        image_show('image',image,1)
        color_overlay = mask_to_color_overlay(truth_mask)
        image_show('color_overlay',color_overlay,1)


        image_show('image1',image1,1)
        #image_show('image2',image2,1)
        color_overlay1 = mask_to_color_overlay(truth_mask1)
        image_show('color_overlay1',color_overlay1,1)


        cv2.waitKey(0)

        continue






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset_reader()

    print( 'sucess!')












