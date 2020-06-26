from torchvision import transforms as tfs
from PIL import Image
from tqdm import *

def processed_img(img):
    '''
    定义图片数据处理函数
    :param img: 传入的图片
    :return:
    '''
    img = tfs.RandomCrop((660,660))(img)
    img = tfs.RandomHorizontalFlip(p=0.5)(img)
    img = tfs.RandomVerticalFlip(p=0.5)(img)
    img = tfs.Resize((96,96))(img)
    return img

img = Image.open("./examples/6.jpg")
img1 = processed_img(img)
# img1.save("./Textures/textures/3.jpg")
# print(img.size)
a=0
for i in tqdm(range(16000)):
    temp_img = processed_img(img)
    temp_img.save("./Textures/textures/"+str(a)+'.jpg')
    a = a+1