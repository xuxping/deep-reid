import tornado.web
import os
import shutil
from utils import photo
from detect import PersonDetect
from infer import Classify

class IndexHandler(tornado.web.RequestHandler):
    """
     Home page for user,photo feeds 主页----所关注的用户图片流
    """
    def get(self,*args,**kwargs):
        self.render('index.html') #打开index.html网页


class ExploreHandler(tornado.web.RequestHandler):
    """
    Explore page,photo of other users 发现页-----发现或最近上传的图片页面
    """
    def get(self,*args,**kwargs):
        # image_urls = get_images("./static/uploads")  #打开指定路径下的文件，或者static/uploads
        os.chdir('static')  # 用于改变当前工作目录到指定的路径
        image_urls = photo.get_images("uploads/thumbs")
        result_urls_p = photo.get_images2("uploads/result")
        os.chdir("..")
        self.render('explore.html',image_urls=image_urls,result_urls_p=result_urls_p, name="prepared")

class PostHandler(tornado.web.RequestHandler):
    """
    Single photo page and maybe  单个图片详情页面
    """
    def get(self,post_id):
        print(post_id)
        self.render('post.html',post_id = post_id)   #根据正则输入的内容，接收到，打开相应的图片


class UploadHandler(tornado.web.RequestHandler):  #上传文件

    # 清空thumbs 和 result 下的文件
    def get(self,*args,**kwargs):
        self.render('upload.html')


    def post(self,*args,**kwargs):
        if (os.path.exists('static/uploads')):
            shutil.rmtree('static/uploads')
        os.mkdir('static/uploads')
        file_imgs = self.request.files.get('newImg',None)  #获取上传文件数据，返回文件列表
        # 显示上传的 query 图片
        for file_img in file_imgs: #可能同一个上传的文件会有多个文件，所以要用for循环去迭代它
            # filename 文件的实际名字，body 文件的数据实体；content_type 文件的类型。 这三个对象属性可以像字典一样支持关键字索引
            save_to = 'static/uploads/{}'.format(file_img['filename'])
            #以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
            with open(save_to,'wb') as f: #二进制
                f.write(file_img['body'])
            # photo.make_thumb(save_to) #同时生成缩略图

            # 调用1）行人检测模型方法 参数： save_to 要查询的图片地址
            detect = PersonDetect()
            imlist = [save_to]
            path = detect.detect(imlist)
            # 调用2）行人查询接口 返回结果图片到 result 下
            classify = Classify()
            img_paths = path
            image_urls = photo.get_images("static/uploads")
            result_urls_p = photo.get_images2(classify.infer(img_paths)[0] + "/"+path[0].split('/')[-1])
        self.render('explore.html', image_urls=image_urls,result_urls_p=result_urls_p)
        # self.redirect('/explore')