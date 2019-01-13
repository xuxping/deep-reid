import glob,os
from PIL import Image

def get_images(path): #将某个路径下所有的文件会返回成一个列表,glob支持*?[]这三种通配符

    #glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）
    image_urls = glob.glob(path + '/*.jpg') #所有path目录下面为*.jpg的图片路径
    return image_urls

def get_images2(path): #将某个路径下所有的文件会返回成一个列表,glob支持*?[]这三种通配符
    #glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）
    result_urls = glob.glob(path + '/*.jpg') #所有path目录下面为*.jpg的图片路径
    return result_urls

#生成缩略图
def make_thumb(path):
    im = Image.open(path)  #打开图片
    im.thumbnail((200,200)) #thumbnail函数接受一个元组作为参数，分别对应着缩略图的宽高，在缩略时，函数会保持图片的宽高比例。
    name = os.path.basename(path) #返回path最后的文件名。如何path以／或\结尾，那么就会返回空值。
    filename,ext = os.path.splitext(name)  #分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
    #save to filename_200x200.jpg
    im.save('static/uploads/thumbs/{}_{}x{}{}'.format(filename,200,200,ext))