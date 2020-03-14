import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os
from copy import deepcopy
from matplotlib import pyplot as plt
import skimage
import os
import skimage.io as io
from skimage import color
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import plot_model
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
img3=Image.open("./dataset/trainy/Y_img_2.bmp")

arr=np.array(img3)
print(arr.shape)

plt.imshow(arr)

plt.show()
