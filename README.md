# image-video-recognition-pytorch

#### 介绍
通过pytorch预训练模型训练自己的分类网络，并分别用本地图片和摄像头检测

#### 环境
python 3.6.4
pytorch 1.3.1


#### 使用说明

1.  data文件夹存放训练数据，数据可以在[这里](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载
2.  model文件夹存放自己训练的模型
3.  train.py文件用来训练自己的模型
4.  predicte_img.py文件可以用来预测本地图片类别
5.  predicte_camera.py可以调用计算机摄像头预测摄像头拍摄到的物体类别

#### 预测结果
训练数据集共两种类别：蚂蚁和蜜蜂，分别用本地图片和摄像头来验证模型训练效果
![本地图片预测结果](https://images.gitee.com/uploads/images/2020/0520/122034_b7da3794_7573147.png "image_predition.png")
![摄像头预测结果](https://gitee.com/datatomoto/image-video-recognition-pytorch/raw/master/video_predition.png)
*注：由于训练EPOCH较少，所以分类的准确率不是很高

#### Reference

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
