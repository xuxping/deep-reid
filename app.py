import tornado.ioloop   #开启循环，让服务一直等待请求的到来
import tornado.web      #web服务基本功能都封装在此模块中
import tornado.options  #从命令行中读取设置
from tornado.options import define,options  #导入包

import webHandler


define('port',default='8000',help='Listening port',type=int) #定义如何接受传进来的东西


class Application(tornado.web.Application):  #引入Application类，重写方法，这样做的好处在于可以自定义，添加另一些功能
    def __init__(self):
        handlers = [
            (r'/', webHandler.IndexHandler),
            (r'/explore', webHandler.ExploreHandler),
            (r'/post/(?P<post_id>[0-9]+)', webHandler.PostHandler), #命名组写法,使用关键字，路由与handler方法不一定顺序一致
            (r'/upload', webHandler.UploadHandler),
            (r'/changeImg', webHandler.changeImgHandler),
        ]
        settings = dict(
            debug = True, #调试模式，修改后自动重启服务，不需要自动重启，生产情况下切勿开启，安全性
            template_path = 'templates', #模板文件目录,想要Tornado能够正确的找到html文件，需要在 Application 中指定文件的位置
            static_path = 'static'  #静态文件目录,可用于用于访问js,css,图片之类的添加此配置之后，tornado就能自己找到静态文件
        )

        super(Application,self).__init__(handlers,**settings) #用super方法将父类的init方法重新执行一遍，然后将handlers和settings传进去，完成初始化


app = Application() #实例化



if __name__ == '__main__':   #当.py文件被直接运行时，代码块将被运行；当.py文件以模块形式被导入时，代码块不被运行。

    tornado.options.parse_command_line()
    app.listen(options.port)  ##如果一个与define语句中同名的设置在命令行中被给出，那么它将成为全局的options的一个属性 即 options.port 相当于define的url的port
    print("Server start on port {}".format(str(options.port)))  #提示服务启动占用端口
    tornado.ioloop.IOLoop.current().start()   #执行ioloop