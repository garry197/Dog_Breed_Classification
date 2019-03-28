# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:29:40 2019

@author: Garry
"""

import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
def load_dataset(path):
  data=load_files(path)
  dog_files=np.array(data['filenames'])
  dog_target=np_utils.to_categorical(np.array(data['target']),133)
  return dog_files,dog_target

train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))



from keras.applications.resnet50 import ResNet50
ResNet50_model = ResNet50(weights='imagenet')


from keras.preprocessing import image                  
from tqdm import tqdm
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


          
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True)
datagen.fit(train_tensors)



import numpy as np
bottleneck_features = np.load('DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet_model.add(Dense(133, activation='softmax'))
ResNet_model.summary()

from keras.optimizers import Adam, Adamax
ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='weights.best_adamax.ResNet50.h5', 
                               verbose=1, save_best_only=True)
epochs = 20
batch_size = 16
ResNet_model.fit(train_ResNet50, train_targets, 
          validation_data=(valid_ResNet50, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)


opt = Adamax(lr=0.0002)
epochs = 10
batch_size = 16

ResNet_model.fit(train_ResNet50, train_targets, 
          validation_data=(valid_ResNet50, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)




ResNet50_predictions = [np.argmax(ResNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]
test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

##for testing purposes
import cv2
import matplotlib.pyplot as plt
def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', pooling='avg',include_top=False).predict(preprocess_input(tensor))

def ResNet50_predict_breed(img_path):
    imge = extract_Resnet50(path_to_tensor(img_path))
    imge = np.expand_dims(imge, axis=0)
    imge = np.expand_dims(imge, axis=0)
    predicted_vector = ResNet_model.predict(imge)
    breed = dog_names[np.argmax(predicted_vector)]
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_rgb)
    return print("The breed of dog is a {}".format(breed))

ResNet50_predict_breed('dogImages/test/022.Belgian_tervuren/Belgian_tervuren_01588.jpg')


