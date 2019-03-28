# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:11:24 2019

@author: Garry
"""

##Loading Dataset
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


#Loading VGG16 pre trained model
import torch
import torchvision.models as models
vgg16=models.vgg16(pretrained=True)

# move model to GPU if CUDA is available
'''use_cuda = torch.cuda.is_available()
if use_cuda:
    vgg16 = vgg16.cuda()
'''
vgg16.cpu()
#preprocessing
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from keras.applications.resnet50 import preprocess_input

dog_files = np.array(glob("dogImages/*/*/*"))

def vgg16_prediction(img_path,model):
  img=Image.open(img_path)
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  preprocess = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize])
  img_tensor = preprocess(img).float()   
  img_tensor.unsqueeze_(0)
  img_tensor = Variable(img_tensor) 
  model.eval()
  output=model(img_tensor)
  output=output.cpu()
  return output.data.numpy().argmax()



def dog_detector(img_path,model):
    y=vgg16_prediction(img_path,model)
    return ((y <= 268) & (y >= 151)) 

#testing the above functions
dog_files_short = dog_files[:300]
dog_detections = np.sum([dog_detector(img, vgg16) for img in tqdm(dog_files_short)])
print('dog detection in dog image set = {}%'.format(dog_detections/len(dog_files_short)))

###WE HAVE USED VGG16 MODEL TO DETECT THE DOG
###NOW WE WILL TRY TO DETECT BREEDS
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch.autograd import Variable
import random
import os
import numpy as np
import time
import copy
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()




data_transforms = {
    'train': transforms.Compose([
#        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),    
}

data_dir = "dogImages/" 
batch_size = 32
num_workers = 0
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}

loaders_data = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                                              shuffle = True, num_workers = num_workers)
                  for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes
n_classes = len(class_names)
  

print(f"No. of Training Records: {dataset_sizes['train']}")
print(f"No. of Validation Records: {dataset_sizes['valid']}")
print(f"No. of Testing Records: {dataset_sizes['test']}")      
print(f"No. of Classes: {n_classes}")
      
use_cuda = torch.cuda.is_available()

model_transfer = torchvision.models.densenet121(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False
num_ftrs = model_transfer.classifier.in_features
model_transfer.classifier = nn.Linear(num_ftrs, n_classes)

if use_cuda:
    model_transfer.cuda()
print(model_transfer)

criterion_transfer = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001, momentum=0.9)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


n_epochs = 15
loaders_transfer = loaders_data
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)


##NOW WE WILL IMPLEMENT THE FUNCTION
import matplotlib.image as mpimg

def predict_breed_transfer(img_path):
    
    img = Image.open(img_path) 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize])
    img_tensor = preprocess(img).float()
    img_tensor.unsqueeze_(0)
    img_tensor = Variable(img_tensor) 
    if use_cuda:
        img_tensor = Variable(img_tensor.cuda())        
    model_transfer.eval()
    output = model_transfer(img_tensor)
    output = output.cpu()
    predict_index = output.data.numpy().argmax() # Our prediction will be the index of the class label with the largest value.
    return predict_index, class_names[predict_index], image_datasets['train'].classes[predict_index]

def display_predictions(img_path):
    #print (img_path)
    pred_index, breed, name = predict_breed_transfer(img_path)
    breed=breed.split('.')[1]
    print("Hey DOG... What's up?!")
        
    # display test image
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(1,2,1)
    img = mpimg.imread(img_path)
    ax.imshow(img)
    plt.axis('off')
    print(f"Predicted Breed: {breed}\n")
    # display sample of matching breed images
    subdir = '/'.join(['dogImages/valid', str(name)])
    file = random.choice(os.listdir(subdir))
    path = '/'.join([subdir, file])
    ax = fig.add_subplot(1,2,2)
    img = mpimg.imread(path)
    ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
    plt.title(breed)
    plt.axis('off')
    plt.show()   
    
    # extract breed from image path
    actual_breed = img_path.split('\\')[1].split('.')[1]
    print(f"Actual Breed: {actual_breed}\n")
    
    print("\n"*3)



test_img_paths = sorted(glob('dogImages/test/*/*'))
np.random.shuffle(test_img_paths)
test_img_paths[1:5]

for img_path in test_img_paths[0:5]:
    display_predictions(img_path)

#display_predictions('dogImages/tes.jpg')