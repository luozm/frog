import os
import numpy as np
import random
import torch
from datetime import datetime

# edit settings here
ROOT_DIR = '/media/lzm/DC9458A6945884C4/projects/frog/'
DATA_DIR = ROOT_DIR + 'data/'
DOWNLOAD_DIR = DATA_DIR + '__download__/'
IMAGE_DIR = DATA_DIR + 'image/'
SPLIT_DIR = DATA_DIR + 'split/'
RESULTS_DIR = ROOT_DIR + 'results/'


# project info
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ---------------------------------------------------------------------------------
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------------
# Print & setup common settings
# ---------------------------------------------------------------------------------

print('@%s:  ' % os.path.basename(__file__))

# setup random seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print('\tset random seed')
print('\t\tSEED=%d' % SEED)


# check GPU & CUDA info

# uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print('\tset cuda environment')
print('\t\ttorch.__version__              =', torch.__version__)
print('\t\ttorch.version.cuda             =', torch.version.cuda)
print('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
try:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
    NUM_CUDA_DEVICES = 1

print('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
print('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())
