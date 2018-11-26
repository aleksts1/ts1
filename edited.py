#%%
# necessary imports
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import SGD, Adam
import pprint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%%
imgHeight, imgWidth = 96, 96
trainDir = "D:/TS1/Cancer/train/"
evaluationDir = "D:/TS1/Cancer/test/"
BATCHSIZE = 32
EPOCHS = 1

# open csv
train_df = pd.read_csv("D:/TS1/Cancer/train_labels.csv")
print('training:')
print(train_df.head())

id_label_map = {k:v for k,v in zip(train_df.id.values, train_df.label.values)}
print(id_label_map)
#%% 
# load files
labeled_files = glob('./train/*.tif')
test_files = glob('./test/*.tif')

print("Train files #: {}".format(len(labeled_files)))
print("Evaluation files #: {}".format(len(test_files)))

#%%
# data generator

datagen = ImageDataGenerator(
    rescale=1./255)

print("[INFO] train generator..")
train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory="./train/",
                                                x_col='id',
                                                y_col='label',
                                                has_ext=False,
                                                class_mode='binary',
                                                target_size=(imgHeight,imgWidth),
                                                batch_size=BATCHSIZE)
print((len(train_generator[3])))                                               
