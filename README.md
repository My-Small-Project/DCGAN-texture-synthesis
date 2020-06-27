# DCGAN
使用DCGAN实现纹理多样性

## Environment

`pytorch >= 1.0`

`torchvision >= 0.3`

## Project structure
`-model.py：模型定义`

`-pre_process.py：数据预处理及训练集生成`

`-train.py：训练模型`

## Dataset

- 数据集开可从百度网盘下载
- 数据集链接：[dataset](https://pan.baidu.com/s/1Uuhj9l61TMW_zj7Qe_tRGw)
- 提取密码：sfb8

```
# 数据集总共有7个，解压压缩包后，可以看到三个目录
- data：存放训练所需要的数据集
- model：存放已经训练好的模型(.pth)
- result：存放已经训练好的结果(.png)
```

## Start
- 在项目下创建Textures文件夹，并且在Textures文件夹中创建textures文件夹，形成目录结构为./Textures/textures
- 在根目录下创建imgs文件夹，形成的目录结构为./imgs
- 将从百度网盘下载到的图像数据集（大小为96*96），或者自己的数据集，放入新建的文件夹中
- run train.py，生成的模型会放在imgs文件夹中
