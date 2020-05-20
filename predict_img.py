# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

class_names = os.listdir('data/train')

img_path = 'test1.jpg'
model_path = 'model/weights.pth'

def predict(image, model, class_names):
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = t(image)
    img = torch.unsqueeze(img, 0)
    output = model_ft(img)
    confidence, preds = torch.max(output, 1)
    return confidence.item(), class_names[int(preds)]
    
def plot_predict(img, confidence, preds):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title('predict:{}, confidence:{:.4f}'.format(preds, confidence))


# build model
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2),
    nn.Softmax()
)
model_ft.load_state_dict(torch.load(model_path))

image = Image.open('test1.jpg')

confidence, preds = predict(image, model_ft, class_names)
print('The input image is: {}, confidence is: {:.4f}'.format(preds, confidence))
plot_predict(image, confidence, preds)