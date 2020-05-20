# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os
import time

class_names = os.listdir('data/train')

img_path = 'test.jpg'
model_path = 'model/weights.pth'

t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# build model
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2),
    nn.Softmax()
)
model_ft.load_state_dict(torch.load(model_path))

# 调用摄像头
capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")
fps = 0.0

while(True):
    t1 = time.time()
    ref,frame=capture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)   
    frame = Image.fromarray(np.uint8(frame))   #将array格式转化为Image格式

    plt.imshow(frame)
    frame_t = t(frame)
    frame_t = torch.unsqueeze(frame_t, 0)
    output = model_ft(frame_t)
    confidence, preds = torch.max(output, 1)
    preds_name = class_names[int(preds)]
    
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print('The predition is :{}, confidence is {:.4f}'.format(preds_name, confidence.item()))
    print("fps= %.2f"%(fps))

    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    
    
    frame = cv2.putText(frame, 'The predition is :{}, confidence is {:.4f}'.format(preds_name, confidence.item()), 
                                   (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("video",frame)
    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break