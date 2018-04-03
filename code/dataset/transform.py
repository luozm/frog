import os
import cv2
import math
import random
import skimage
import skimage.morphology
from PIL import Image
import numpy as np


## for debug
def dummy_transform(image):
    print ('\tdummy_transform')
    return image

# kaggle science bowl-2 : ################################################################

def resize_to_factor2(image, mask, factor=16):

    H,W = image.shape[:2]
    h = (H//factor)*factor
    w = (W //factor)*factor
    return fix_resize_transform2(image, mask, w, h)



def fix_resize_transform2(image, mask, w, h):
    H,W = image.shape[:2]
    if (H,W) != (h,w):
        image = cv2.resize(image,(w,h))

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
    return image, mask




def fix_crop_transform2(image, mask, x,y,w,h):

    H,W = image.shape[:2]
    assert(H>=h)
    assert(W >=w)

    if (x==-1 & y==-1):
        x=(W-w)//2
        y=(H-h)//2

    if (x,y,w,h) != (0,0,W,H):
        image = image[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    return image, mask


def random_crop_transform2(image, mask, w,h, u=0.5):
    x,y = -1,-1
    if random.random() < u:

        H,W = image.shape[:2]
        if H!=h:
            y = np.random.choice(H-h)
        else:
            y=0

        if W!=w:
            x = np.random.choice(W-w)
        else:
            x=0

    return fix_crop_transform2(image, mask, x,y,w,h)



def random_horizontal_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
        mask  = cv2.flip(mask,1)
    return image, mask

def random_vertical_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)
    return image, mask

def random_rotate90_transform2(image, mask, u=0.5):
    if random.random() < u:

        angle=random.randint(1,3)*90
        if angle == 90:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,1)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        elif angle == 180:
            image = cv2.flip(image,-1)
            mask = cv2.flip(mask,-1)

        elif angle == 270:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,0)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,0)
    return image, mask


def relabel_multi_mask(multi_mask):
    data = multi_mask
    data = data[:,:,np.newaxis]
    unique_color = set( tuple(v) for m in data for v in m )
    #print(len(unique_color))


    H,W  = data.shape[:2]
    multi_mask = np.zeros((H,W),np.int32)
    for color in unique_color:
        #print(color)
        if color == (0,): continue

        mask = (data==color).all(axis=2)
        label  = skimage.morphology.label(mask)

        index = [label!=0]
        multi_mask[index] = label[index]+multi_mask.max()

    return multi_mask


def random_shift_scale_rotate_transform2( image, mask,
                        shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
                        rotate_limit=[-15,15], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):

    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        sx    = scale
        sy    = scale
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        mask = mask.astype(np.float32)
        mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = mask.astype(np.int32)
        mask = relabel_multi_mask(mask)

    return image, mask




# single image ########################################################

#agumentation (photometric) ----------------------
def random_brightness_shift_transform(image, limit=[16,64], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = image + alpha*255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_brightness_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_contrast_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha*image + gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_saturation_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef  = np.array([[[0.114, 0.587,  0.299]]])
        gray  = image * coef
        gray  = np.sum(gray,axis=2, keepdims=True)
        image = alpha*image  + (1.0 - alpha)*gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue_transform(image, limit=[-0.1,0.1], u=0.5):
    if random.random() < u:
        h = int(np.random.uniform(limit[0], limit[1])*180)
        #print(h)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def random_noise_transform(image, limit=[0, 0.5], u=0.5):
    if random.random() < u:
        H,W = image.shape[:2]
        noise = np.random.uniform(limit[0],limit[1],size=(H,W))*255

        image = image + noise[:,:,np.newaxis]*np.array([1,1,1])
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# geometric ---
def resize_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = (height//factor)*factor
    w = (width //factor)*factor
    return fix_resize_transform(image, w, h)


def fix_resize_transform(image, w, h):
    height,width = image.shape[:2]
    if (height,width) != (h,w):
        image = cv2.resize(image,(w,h))
    return image


def pad_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = math.ceil(height/factor)*factor
    w = math.ceil(width/factor)*factor

    image = cv2.copyMakeBorder(image, top=0, bottom=h-height, left=0, right=w-width,
                               borderType=cv2.BORDER_REFLECT101, value=[0, 0, 0])

    return image


class GaussianDistortion:
    """
    This class performs randomised, elastic gaussian distortions on images.
    """
    def __init__(self, probability, grid_width, grid_height, magnitude, corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05, sdy=0.05):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float
        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:
        .. math::
         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True
        self.corner = corner
        self.method = method
        self.mex = mex
        self.mey = mey
        self.sdx = sdx
        self.sdy = sdy

    def perform_operation(self, images):
        """
        Distorts the passed image(s) according to the parameters supplied
        during instantiation, returning the newly distorted image.
        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(math.floor(w / float(horizontal_tiles)))
        height_of_square = int(math.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def sigmoidf(x, y, sdx=0.05, sdy=0.05, mex=0.5, mey=0.5, const=1):
            sigmoid = lambda x1, y1:  (const * (math.exp(-(((x1-mex)**2)/sdx + ((y1-mey)**2)/sdy) )) + max(0,-const) - max(0, const))
            xl = np.linspace(0,1)
            yl = np.linspace(0, 1)
            X, Y = np.meshgrid(xl, yl)

            Z = np.vectorize(sigmoid)(X, Y)
            mino = np.amin(Z)
            maxo = np.amax(Z)
            res = sigmoid(x, y)
            res = max(((((res - mino) * (1 - 0)) / (maxo - mino)) + 0), 0.01)*self.magnitude
            return res

        def corner(x, y, corner="ul", method="out", sdx=0.05, sdy=0.05, mex=0.5, mey=0.5):
            ll = {'dr': (0, 0.5, 0, 0.5), 'dl': (0.5, 1, 0, 0.5), 'ur': (0, 0.5, 0.5, 1), 'ul': (0.5, 1, 0.5, 1), 'bell': (0, 1, 0, 1)}
            new_c = ll[corner]
            new_x = (((x - 0) * (new_c[1] - new_c[0])) / (1 - 0)) + new_c[0]
            new_y = (((y - 0) * (new_c[3] - new_c[2])) / (1 - 0)) + new_c[2]
            if method == "in":
                const = 1
            else:
                if method == "out":
                    const =- 1
                else:
                    const = 1
            res = sigmoidf(x=new_x, y=new_y,sdx=sdx, sdy=sdy, mex=mex, mey=mey, const=const)

            return res

        def do(image):

            for a, b, c, d in polygon_indices:
                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]

                sigmax = corner(x=x3/w, y=y3/h, corner=self.corner, method=self.method, sdx=self.sdx, sdy=self.sdy, mex=self.mex, mey=self.mey)
                dx = np.random.normal(0, sigmax, 1)[0]
                dy = np.random.normal(0, sigmax, 1)[0]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    from dataset.reader import ScienceDataset
    import matplotlib.pyplot as plt
    from skimage import exposure
    from common import IMAGE_DIR
    # import Augmentor
    # p = Augmentor.Pipeline(IMAGE_DIR + 'train1_norm/images')
    # p.random_distortion(0.5,16,16,5)
    # img = p.sample(10)
    # plt.imshow(img)
    # plt.show()

    dataset = ScienceDataset('train1_train_603', img_folder='train1_norm', mask_folder='stage1_train', mode='train')

    for img, mask, meta, idx in dataset:
        mask_re = mask.reshape(mask.shape + (1,))
        img_concat = np.stack((img[:, :, 0], mask), axis=2)
        result = random_contrast_transform(img, u=1)
        result2 = random_brightness_transform(img, u=1)

        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)



        # result = elastic_transform_2(img[:,:,0],img.shape[1]*2, img.shape[1]*0.08)
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_adapteq)
        plt.figure()
        plt.imshow(result2)
        plt.figure()
        plt.imshow(img_rescale)
        plt.show()
#        elastic_transform(img_concat, img_concat.shape[1]*2, img_concat.shape[1]*0.08, img_concat.shape[1]*0.08)
    print()

    print('\nsucess!')

