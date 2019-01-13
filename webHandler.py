import tornado.web
import os
import shutil
from utils import photo
from detect import PersonDetect
from infer import Classify
import json

detect = PersonDetect(dist='static/person_detect')

classify = Classify(save_dir='static/')


class IndexHandler(tornado.web.RequestHandler):
    """
     Home page for user,photo feeds 主页----所关注的用户图片流
    """

    def get(self, *args, **kwargs):
        self.redirect('/upload')
        # self.render('index.html')  # 打开index.html网页


class ExploreHandler(tornado.web.RequestHandler):
    """
    Explore page,photo of other users 发现页-----发现或最近上传的图片页面
    """

    def get(self, *args, **kwargs):
        # image_urls = get_images("./static/uploads")  #打开指定路径下的文件，或者static/uploads
        os.chdir('static')  # 用于改变当前工作目录到指定的路径
        image_urls = photo.get_images("uploads/thumbs")
        result_urls_p = photo.get_images2("uploads/result")
        os.chdir("..")
        self.render('explore.html', image_urls=image_urls, result_urls_p=result_urls_p, name="prepared")


class PostHandler(tornado.web.RequestHandler):
    """
    Single photo page and maybe  单个图片详情页面
    """

    def get(self, post_id):
        print(post_id)
        self.render('post.html', post_id=post_id)  # 根据正则输入的内容，接收到，打开相应的图片


class changeImgHandler(tornado.web.RequestHandler):  # 点击行人监测后的文件

    # 清空thumbs 和 result 下的文件
    def get(self, *args, **kwargs):
        if (os.path.exists('static/uploads')):
            shutil.rmtree('static/uploads')
        os.mkdir('static/uploads')
        filename = self.get_argument('filename')
        # filename = self.request.arguments.get('filename')
        # image_urls = self.request.arguments.get('image_urls')
        # dect_img_paths = self.request.arguments.get('dect_img_paths')
        # filename 文件的实际名字，body 文件的数据实体；content_type 文件的类型。 这三个对象属性可以像字典一样支持关键字索引
        save_to = 'static/person_detect/{}'.format(filename)
        # 调用2）行人查询接口 返回结果图片到 result 下
        result_urls_p = classify.infer([save_to])
        # 行人检索结果
        result_urls_p = photo.get_images2(os.path.join(result_urls_p[0],filename))

    def post(self, *args, **kwargs):
        if (os.path.exists('static/uploads')):
            shutil.rmtree('static/uploads')
        os.mkdir('static/uploads')
        filename = self.get_argument('filename')
        # image_urls = self.get_argument.get('image_urls')
        # dect_img_paths = self.get_argument.get('dect_img_paths')
        # filename 文件的实际名字，body 文件的数据实体；content_type 文件的类型。 这三个对象属性可以像字典一样支持关键字索引
        save_to = 'static/person_detect/{}'.format(filename)
        # 调用2）行人查询接口 返回结果图片到 result 下
        # result_urls_p = classify.infer([save_to])
        # 行人检索结果
        result_urls_p = photo.get_images2(os.path.join("static/ranked_results", filename))
        self.write(json.dumps(dict(image_urls="", dect_img_paths="", result_urls_p=result_urls_p)))
        # self.render('upload.html', image_urls='', dect_img_paths='', result_urls_p=result_urls_p)

class UploadHandler(tornado.web.RequestHandler):  # 上传文件

    # 清空thumbs 和 result 下的文件
    def get(self, *args, **kwargs):
        image_urls = ''
        result_urls_p = ''
        dect_img_paths = ''
        self.render('upload.html', image_urls=image_urls, dect_img_paths=dect_img_paths, result_urls_p=result_urls_p)

    def post(self, *args, **kwargs):
        if (os.path.exists('static/uploads')):
            shutil.rmtree('static/uploads')
        os.mkdir('static/uploads')
        file_imgs = self.request.files.get('newImg', None)  # 获取上传文件数据，返回文件列表
        # 显示上传的 query 图片
        for file_img in file_imgs:  # 可能同一个上传的文件会有多个文件，所以要用for循环去迭代它
            # filename 文件的实际名字，body 文件的数据实体；content_type 文件的类型。 这三个对象属性可以像字典一样支持关键字索引
            save_to = 'static/uploads/{}'.format(file_img['filename'])
            # 以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
            # 如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
            with open(save_to, 'wb') as f:  # 二进制
                f.write(file_img['body'])
            # 调用1）行人检测模型方法 参数： save_to 要查询的图片地址
            imlist = [save_to]
            # 行人监测结果
            dect_img_paths = detect.detect(imlist)
            # 要检索的图片
            image_urls = photo.get_images("static/uploads")
            # 调用2）行人查询接口 返回结果图片到 result 下
            infrepath = classify.infer(dect_img_paths)
            imgpath = infrepath[0] + "/" + dect_img_paths[0].split('/')[-1]
            # 行人检索结果
            result_urls_p = photo.get_images2(imgpath)
        self.render('upload.html', image_urls=image_urls, dect_img_paths=dect_img_paths, result_urls_p=result_urls_p)
