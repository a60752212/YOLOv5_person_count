import traceback
from unittest import result
from UI.ui import Ui_MainWindow
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
# from utils.datasets import IMG_FORMATS, VID_FORMATS
import counter
import os
import numpy as np

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# 预测线程, 调用yolo模型进行推理
class PredictThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    trigger = pyqtSignal(int)

    def __init__(self, mainwin):
        # 初始化函数
        # super().__init__()
        super(PredictThread, self).__init__()
        self.mainwin = mainwin

    def run(self):
        #重写线程执行的run函数
        try:
            counter.run(self.mainwin.source, counter.getframe(self.mainwin.source), self.mainwin.c1, self.mainwin.c2, self.mainwin)
            #触发自定义信号
            self.trigger.emit(1)
            # self.mainwin.model_class.run(self.mainwin.source, False, '')
        except Exception as e:
            # print('ERROR: %s' %(e))
            print(traceback.print_exc())

class MainWin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("标题")
        desktop = QApplication.desktop()
        min_ratio = 0.5
        max_ratio = 0.8
        self.setGeometry(int((1-max_ratio)*desktop.width()/2), int((1-max_ratio)*desktop.height()/2), int(max_ratio*desktop.width()), int(max_ratio*desktop.height()))


        # connect 方法可以设置对应控件(按钮等)调用哪个函数
        self.PB_import.clicked.connect(self.importMedia) # 打开按钮
        self.PB_predict.clicked.connect(self.run) # 预测按钮
        self.PB_predict.setEnabled(False) 
        self.PB_stop.clicked.connect(self.stopPredict) # 停止按钮
        self.stop = 0 # 视频/摄像头时起作用, 当stop == 1时, 视频/摄像头停止播放
        # 菜单栏
        self.PB_c1.clicked.connect(self.selectC1)  # 选择区域1
        self.PB_c2.clicked.connect(self.selectC2)  # 选择区域2

        self.canrun = 0
        self.source = None

        self.predictThread = PredictThread(self)  # 初始化预测线程
        self.predictThread.trigger.connect(self.isdone)

        self.c1 = None
        self.c2 = None

        self.num_thres = 5
        self.lineEdit.setText(str(self.num_thres))
        self.PB_setThres.clicked.connect(self.setThres)

    def setThres(self):
        self.num_thres = int(self.lineEdit.text())

    def resizeImg(self, image):
        '''
        调整图片到合适大小
        '''
        width = image.width()  ##获取图片宽度
        height = image.height() ##获取图片高度
        if width / self.labelsize[1] >= height / self.labelsize[0]: ##比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.labelsize[1]
        else:
            ratio = height / self.labelsize[0]
        new_width = width / ratio  ##定义新图片的宽和高
        new_height = height / ratio
        new_img = image.scaled(new_width, new_height)##调整图片尺寸
        return new_img

    def padding(self, image):
        '''
        图片周围补0以适应label大小
        '''
        width = image.shape[1]
        height = image.shape[0]
        target_ratio = self.labelsize[0]/self.labelsize[1] # h/w
        now_ratio = height/width
        if target_ratio>now_ratio:
            # padding h
            new_h = int(target_ratio*width)
            padding_image = np.ones([int((new_h-height)/2), width, 3], np.uint8)*255
            new_img = cv2.vconcat([padding_image, image, padding_image])
        else:
            # padding w
            new_w = int(height/target_ratio)
            padding_image = np.ones([height, int((new_w-width)/2), 3], np.uint8)*255
            new_img = cv2.hconcat([padding_image, image, padding_image])
        new_img = cv2.resize(new_img, (self.labelsize[1], self.labelsize[0]))
        return new_img

    def resize_label(self):
        '''
        更新label中的图片大小
        '''
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        img_in = self.label_in.pixmap()
        img_out = self.label_out.pixmap()
        try:
            img_in = self.resizeImg(img_in)
        except:
            return
        else:
            self.label_in.setPixmap(img_in)

        try:
            img_out = self.resizeImg(img_out)
        except:
            return
        else:    
            self.label_out.setPixmap(img_out)

    def getURL(self):
        text, ok=QInputDialog.getText(self, 'Text Input Dialog', '输入链接')
        is_url = text.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if ok and is_url:
            self.GetstrlineEdit.setText(str(text))
            self.source = text
            print(self.source)

    def importMedia(self):
        '''
        打开检测源
        '''
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        # 源为摄像头
        if self.RB_camera.isChecked():
            self.source = 0
            frame = counter.getframe(self.source)
            # cap = cv2.VideoCapture(0)
            # # 获取视频宽度
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # # 获取视频高度
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # frame = np.ones([frame_height, frame_width, 3], np.uint8)*255
            QMessageBox.information(self, "信息提示框", "请依次选择两个区域！", QMessageBox.Yes)
            self.c1 = counter.zuobiao(frame)
            cv2.polylines(frame,[np.array(self.c1,np.int32)],True,(0,0,255),2,8,0)# 绘制多边形
            self.c2 = counter.zuobiao(frame)
            QMessageBox.information(self, "信息提示框", "打开摄像头中！", QMessageBox.Yes)
            self.run()
            # print('<font color=green>请载入模型进行预测...</font>')
        # 源为图片/视频
        if self.RB_url.isChecked():
            self.getURL()
            frame = counter.getframe(self.source)
            # cap = cv2.VideoCapture(self.source)
            # # 获取视频宽度
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # # 获取视频高度
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # frame = np.ones([frame_height, frame_width, 3], np.uint8)*255
            QMessageBox.information(self, "信息提示框", "请依次选择两个区域！", QMessageBox.Yes)
            self.c1 = counter.zuobiao(frame)
            cv2.polylines(frame,[np.array(self.c1,np.int32)],True,(0,0,255),2,8,0)# 绘制多边形
            self.c2 = counter.zuobiao(frame)
            QMessageBox.information(self, "信息提示框", "打开摄像头中！", QMessageBox.Yes)
            self.run()
        elif self.RB_img.isChecked():
            fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            # print(fname)
            if fname.split('.')[-1].lower() in (VID_FORMATS):
                self.importImg(fname)
                self.source = fname          
            else:
                print('<font color=red>不支持该类型文件...</font>')
        else:
            print('<font color=red>请选择检测源类型...</font>')
    
    def importImg(self, file_name):
        '''
        label_in 中显示图片/视频第一帧
        '''
        if file_name.split('.')[-1].lower() in VID_FORMATS:
            cap = cv2.VideoCapture(file_name)
            # print(file_name)
            if cap.isOpened():
                # self.video = True
                # print(111)
                ret, img_in = cap.read()
                if ret:
                    # img_in = self.resizeImg(img_in)
                    img_in = self.padding(img_in)
                    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
                    # padding
                    img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
                    img_in = QPixmap(img_in)
            cap.release()
        elif file_name.split('.')[-1].lower() in IMG_FORMATS:
            # self.video = False
            img_in = cv2.imread(file_name)
            img_in = self.padding(img_in)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
            img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
            img_in = QPixmap(img_in)
        if img_in.isNull():
            print('<font color=red>打开失败...</font>')
            return
        # img_in = img_in.scaledToWidth(self.labelsize[1])
        img_in = self.resizeImg(img_in)
        self.label_in.setPixmap(img_in)

    def stopPredict(self):
        '''
        stop == 1 播放停止
        '''
        self.stop = 1


    def selectC1(self):
        frame = counter.getframe(self.source)
        if self.c2 is not None:
            cv2.polylines(frame,[np.array(self.c2,np.int32)],True,(0,0,255),2,8,0)# 绘制多边形
        self.c1 = counter.zuobiao(frame)
        print(self.c1)
        if self.c1 is not None and self.c2 is not None:
            self.PB_predict.setEnabled(True)
            self.canrun = 1

    def selectC2(self):
        frame = counter.getframe(self.source)
        if self.c1 is not None:
            cv2.polylines(frame,[np.array(self.c1,np.int32)],True,(0,0,255),2,8,0)# 绘制多边形
        self.c2 = counter.zuobiao(frame)
        print(self.c2)
        if self.c1 is not None and self.c2 is not None:
            self.PB_predict.setEnabled(True)
            self.canrun = 1

    def run(self):
        '''
        开始预测
        '''
        self.predictThread.start()
        self.canrun = 0
        self.PB_predict.setEnabled(False)
        self.PB_stop.setEnabled(True)

    def isdone(self, done):
        '''
        结束一次预测
        '''
        if done == 1:
            self.canrun = 1
            self.PB_predict.setEnabled(True)
            self.action_loadmodel.setEnabled(True)
            # self.PB_import.setEnabled(True)
            self.PB_stop.setEnabled(False)
            self.stop = 0
            self.predictThread.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWin()
    main.show()
    sys.exit(app.exec_())