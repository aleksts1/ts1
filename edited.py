#%%
# necessary imports
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
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
testDir = "D:/TS1/Cancer/test/"
BATCHSIZE = 32
EPOCHS = 1

# open csv
train_df = pd.read_csv("D:/TS1/Cancer/train_labels.csv")
test_df = pd.read_csv("D:/TS1/Cancer/sample_submission.csv")
print('training:')
print(train_df.head())

#%% 
# load files
labeled_files = glob(trainDir +'*.tif')
test_files = glob(testDir + '*.tif')

print("Train files #: {}".format(len(labeled_files)))
print("Evaluation files #: {}".format(len(test_files)))

      
      
#%%
# initializing architecture
print("[INFO] creating architecture..")
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(imgWidth, imgHeight, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = Adam(lr=0.001)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
      
#%%
# training data generator

datagen = ImageDataGenerator(
    rescale=1./255)

print("[INFO] train generator..")
train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory=trainDir,
                                                x_col='id',
                                                y_col='label',
                                                has_ext=False,
                                                class_mode='binary',
                                                target_size=(imgHeight,imgWidth),
                                                batch_size=BATCHSIZE)
                                             
#%%
# testing data generator
#testImages = testDir.
print("[INFO] test generator")
test_generator = datagen.flow_from_dataframe(dataframe=test_df,
                                                directory=testDir,
                                                x_col='id',
                                                y_col=None,
                                                has_ext=False,
                                                class_mode=None,
                                                target_size=(imgHeight,imgWidth),
                                                batch_size=BATCHSIZE)
print((test_generator[0].shape))
#%%
# compiling model
H = model.fit_generator(train_generator, epochs=EPOCHS,
                        steps_per_epoch=train_generator.n//BATCHSIZE)


#%%
# predict
pred = model.predict_on_batch(test_generator[0])
print(pred)

#%%
# predict generator
pred_gen = model.predict_generator(test_generator,steps=len(test_generator))
print(pred_gen)

#%%
print(pred_gen[1][0])
#%%
# saving model
print("[INFO] saving model...")

model.save('model.model')        
model.save_weights('first_try.h5')

#%%
# debug 
print(type(pred_gen))

#%%
# merge to dataframe and output as .csv
ids = test_df.id.values
preds = [item[0] for item in pred_gen]
print(len(ids) == len(preds))
df = pd.DataFrame({'id': ids, 'label' : preds})
df.to_csv('C:/Users/xc/Documents/GitHub/ts1/submit_predictions.csv', index = False)
df.head()

#%%
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Cancer challenge")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')