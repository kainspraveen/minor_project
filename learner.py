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
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import plot_model
import os
from tensorflow.keras.optimizers import SGD
from numpy import asarray
import matplotlib.image as mpimg
from skimage import color
from tensorflow.keras import backend as K
from skimage import io
import sys
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

class MModel:




    def __init__(self):
        self.rgb_data=None
        self.label=np.array([])
        self.model=tf.keras.models.Sequential()
        self.hsv_data=None
        self.stacked=None
        self.ground_truth=None
    def DataInput(self):
        img=Image.open("./dataset/trainx/X_img_1.bmp")
        img3=Image.open("./dataset/trainy/Y_img_1.bmp")
        #img3=self.rgb2gray(img3)
        img=img.resize((128,128))
        img3=img3.resize((128,128))

        arr=np.array(img)

        arr3=np.array(img3)
        #print(arr3.shape)
        #arr3=color.rgb2gray(arr3)
        #self.rgb_data=deepcopy(arr)
        #print(type(self.rgb_data))
        #print("_____________________________")
        tup=(arr,)
        tup2=(skimage.color.convert_colorspace(arr,'RGB','HSV'),)
        tup3=(arr3,)
        temp_tup=(np.concatenate((arr,skimage.color.convert_colorspace(arr,'RGB','HSV')), axis=2),)
        #print("_______________",np.array(Image.open("./dataset/trainy/Y_img_1.bmp").resize((128,128))).shape)
        """plt.imshow(arr3,interpolation=None,cmap='gray',vmin=0,vmax=255)
        plt.title("ground truth")
        plt.show()"""
        #print(self.rgb_data.shape)
        print(tup3[0].shape)
        for i in range(2,901):
            loc="./dataset/trainx"
            loc3="./dataset/trainy"
            loc=loc+"/X_img_"+str(i)+".bmp"
            loc3=loc3+"/Y_img_"+str(i)+".bmp"
            img=Image.open(loc)
            img3=Image.open(loc3)
            #img3=self.rgb2gray(img3)
            img=img.resize((128,128))
            img3=img3.resize((128,128))

            arr=np.array(img)
            arr3=np.array(img3)
            #print(arr3.size)
            #print(arr3.size)
            #arr3=color.rgb2gray(arr3)
            arr2=skimage.color.convert_colorspace(arr,'RGB','HSV')
            rand=np.concatenate((arr,arr2), axis=2)
            temp_tup+=(rand,)
            #print("shape",arr.shape)
            #self.rgb_data=np.stack((self.rgb_data,arr),axis=0)
            #print(self.rgb_data.shape)
            tup+=(arr,)
            tup2+=(arr2,)
            tup3+=(arr3,)
            print ("pic....",i)
            print(tup3[1].shape)

        #loc="./dataset/trainx"
        self.rgb_data=np.stack(tup)
        self.hsv_data=np.stack(tup2)
        self.ground_truth=np.stack(tup3)
        self.stacked=np.stack(temp_tup)
        print(self.ground_truth.shape)
        print(self.rgb_data.shape)
        """plt.imshow(self.rgb_data[0],interpolation=None)
        plt.title("rgb")
        plt.show()
        plt.imshow(self.ground_truth[0],interpolation=None,cmap='gray',vmin=0,vmax=255)
        plt.title("ground truth")
        plt.show()"""
        #print("______________________________________")
        #print(self.rgb_data[0,2,99])

    def SegmentationModel(self):
        """ Lookup for integrating HSV matrix along with RGB matrix"""
        """modify Conv2D parameters Later according to the paper"""
        cnn_layer=tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))
        pool=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=1)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(pool)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(pool)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(cnn_layer)
        self.model.add(pool)


    def unet(self,pretrained_weights = None,input_size = (128,128,6)):

        """ modify the output dimensionality of each layer according to the paper"""


        inputs = Input(shape=input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        #conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        #conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        mode = Model(inputs, conv10)

        mode.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['accuracy'])
        #plot_model(mode, to_file="model.png", show_shapes=True)
        mode.summary()
        print("\n\n\n........................................TRAINING STARTS .............................................\n")
        print(self.ground_truth.shape)
        mode.fit(self.stacked[:10],self.ground_truth[:10],epochs=100,verbose=2)
        temp=mode.predict(self.stacked[891:901])
        print(temp[0].shape)
        #plt.imshow(temp[0,:,:,0],interpolation=None)
        #plt.show()

    def unet_alt(self, pretrained_weights = None,input_size = (128,128,6)):
        inputs = Input(shape=input_size)
        conv1 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv3 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv5 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv6 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv6)
        conv7 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv8 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv9 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        """conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)"""
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
        drop4 = Dropout(0.2)(pool4)
        fc1=Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
        drop5=Dropout(0.2)(fc1)
        fc2=Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)

        """conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)"""

        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(fc2))
        merge6 = concatenate([conv9,up6], axis = 3)
        drop6=Dropout(0.2)(merge6)
        conv10 = Conv2D(128, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
        conv11 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
        conv12 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
        up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv12))
        merge7 = concatenate([conv6,up7], axis = 3)
        drop7=Dropout(0.2)(merge7)
        conv13 = Conv2D(64, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
        conv14 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
        conv15 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)
        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv15))
        merge8 = concatenate([conv3,up8], axis = 3)
        drop8=Dropout(0.2)(merge8)
        conv16 = Conv2D(32, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
        conv17 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv16)
        conv18 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv17)
        conv19 = Conv2D(1, 1, activation = 'sigmoid')(conv18)

        mode = Model(inputs, conv19)

        mode.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
        plot_model(mode, to_file="model.png", show_shapes=True)
        mode.summary()
        print("\n\n\n......................................TRAINING STARTS ............................................\n")
        print("ground trtuth shape", self.ground_truth.shape)
        history=mode.fit(self.stacked[:890],self.ground_truth[:890],epochs=100,verbose=1)

        print("\n accuracy values for each epoch\n")
        print(history.history['accuracy'])
        print("\n loss values for each epoch\n")
        print(history.history['loss'])
        print("\n\n")

        plt.plot(history.history['accuracy'],color='red')
        plt.plot(history.history['loss'],color='blue')
        plt.title('model stats for 13 samples')
        plt.ylabel('accuracy/loss')
        plt.xlabel('epoch')
        plt.legend(['accuracy', 'loss'], loc='upper left')
        plt.savefig("model_stat_graph.png")
        plt.show()



        #print("FIRST TRAINING IMAGE", self.rgb_data[0])
        temp=mode.predict(self.stacked[891:901])
        np.save('predicted_array',temp)
        #print("output shape", temp[0,:,:,0].shape)
        #print("FIRST TESTED IMAGE RESULT", temp[0])
        #print("showing ground_truth")
        plt.imshow(temp[0,:,:,0],interpolation=None,cmap='gray',vmin=0,vmax=1)
        plt.title("predicted")
        plt.show()



    def Preprocessing(self):
        """ preprocessing for RGB Data """
        for i in range(len(self.rgb_data)):
            self.rgb_data[i]=self.rgb_data[i].astype('float32')
            mean, std = self.rgb_data[i].mean(), self.rgb_data[i].std()
            self.rgb_data[i]=(self.rgb_data[i]-mean)/std

        """ preprocessing for HSV data """
        for i in range(len(self.hsv_data)):
            self.hsv_data[i]=self.rgb_data[i].astype('float32')
            mean, std = self.hsv_data[i].mean(), self.hsv_data[i].std()
            self.hsv_data[i]=(self.hsv_data[i]-mean)/std

        """ preprocessing for input """
        for i in range(len(self.stacked)):
            self.stacked[i]=self.stacked[i].astype('float32')
            mean, std = self.stacked[i].mean(), self.stacked[i].std()
            self.stacked[i]=(self.stacked[i]-mean)/std

        #print("STACKED DATA", self.stacked[0])

        """ preprocessing for output ground thruth """

        for i in range(len(self.ground_truth)):
            self.ground_truth[i]=self.ground_truth[i].astype('float32')
            self.ground_truth[i]=self.ground_truth[i]/255

        #print("GROUND TRUTH", self.ground_truth[0])


    def ClassificationModel(self, segmented_data):
        pass

    def ConvTest(self,test=None):
        img1=self.hsv_data[5]
        print(img1.shape)
        #print("self.hsv_data[]")
        img=self.rgb_data[5]
        #plt.imshow(img, interpolation='none')
        #plt.show()
        #plt.imshow(img1,interpolation='None')
        #plt.show()
        #model=tf.keras.models.Sequential()
        filter=[[1,0,0],[0,1,0],[0,0,1]]
        filter=np.asarray(filter)
        #out=tf.nn.conv2d(img,filters=filter,strides=1,padding="VALID")
        #model.add(layer)
        print ("_________________________________________________________________")
        #print out
        #image = np.array(out, dtype=np.uint8)[...,::-1]
        #plt.imshow(image_transp, interpolation='none')

model=MModel()
model.DataInput()
model.Preprocessing()
#model.unet()
model.unet_alt()
model.ConvTest()
