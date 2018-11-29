
# coding: utf-8

# In[1]:


# necessary imports
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
import pprint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt


# In[4]:


#%%
imgHeight, imgWidth = 96, 96
trainDir = "./train/"
testDir = "./test/"
BATCHSIZE = 32
EPOCHS = 5

# open csv
train_df = pd.read_csv("train_labels.csv")
test_df = pd.read_csv("sample_submission.csv")
print('Loading .csv:')
print(train_df.head())
print(test_df.head())


# In[5]:


# load files
labeled_files = glob(trainDir +'*.tif')
test_files = glob(testDir + '*.tif')

print("Train files #: {}".format(len(labeled_files)))
print("Evaluation files #: {}".format(len(test_files)))


# In[7]:


#%%
# initializing architecture
print("[INFO] creating architecture..")

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(imgWidth, imgHeight, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

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

print("[INFO] architecture created..")


# In[8]:


# training data generator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    shear_range=15,
    horizontal_flip=True,
    vertical_flip=True,
    )

print("[INFO] train generator..")
train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory=trainDir,
                                                x_col='id',
                                                y_col='label',
                                                has_ext=False,
                                                class_mode='binary',
                                                target_size=(imgHeight,imgWidth),
                                                batch_size=BATCHSIZE)


# In[9]:


print("[INFO] test generator")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                 directory=testDir,
                                                 x_col='id',
                                                 y_col='label',
                                                 has_ext=False,
                                                 #class_mode='binary',
                                                 target_size=(imgHeight,imgWidth),
                                                 batch_size=BATCHSIZE)


# In[10]:


# compiling model
H = model.fit_generator(train_generator,
                        validation_data=test_generator,
                        validation_steps=int(test_generator.n//BATCHSIZE),
                        steps_per_epoch=int(train_generator.n//BATCHSIZE),
                        epochs=EPOCHS)


# In[ ]:


# predict generator
pred_gen = model.predict_generator(test_generator,steps=len(test_generator))
#print(pred_gen)
print(H.history)


# In[ ]:


# saving model
print("[INFO] saving model...")

model.save('model.model')        
model.save_weights('first_try.h5')


# In[ ]:


# merge to dataframe and output as .csv
ids = test_df.id.values
preds = [item[0] for item in pred_gen]
print(len(ids) == len(preds))
df = pd.DataFrame({'id': ids, 'label' : preds})
df.to_csv('submit_predictions.csv', index = False)
df.head()


# In[ ]:


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
plt.savefig('plotCombined.png')


#----------------------
# Plot training & validation accuracy values
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('plot_acc_valacc.png')

# Plot training & validation loss values
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('plot_loss_valloss.png')
#----------------------------------------------

