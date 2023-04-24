import os
import time
import numpy as np

import tracker
from detector import Detector
import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

def getframe(root):
    vc = cv2.VideoCapture(root)  # 读取需要处理的视频
    if vc.isOpened():  # 判读视频是否正常打开
        print("打开ok")
    else:
        print("打开失败，程序退出")
        exit(-1)  # 如果不能正常打开则自动结束程序

    ok, frame = vc.read()  #按帧读取视频，返回两个数值，第一个布尔值,这里只读取一张图片，所以不用循环获取所有帧图片
    vc.release()  # 需要释放
    return frame

def point_in_polygon(x, y, polygon):  #射线法判断点是否在区域内
    num = len(polygon)
    j = num - 1
    c = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            c = not c
        j = i
    return c

def zuobiao(frame):
    img = frame  # 加载本地的图像
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//2, h//2))  #对图像进行缩小便于后续选择坐标点
    a = []  # 用于存放横坐标
    b = []  # 用于存放纵坐标

    print('请随意点击4个坐标：')

    # 定义点击事件
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 如果存在鼠标点击事件
            xy = "%d,%d" % (x, y)  # 得到坐标x,y
            a.append(x*2)  # 将每次的坐标存放在a数组里面 由于原先图片尺寸缩小一半，这里坐标点需要变回原图片的尺寸位置
            b.append(y*2)  # 将每次的坐标存放在b数组里面
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)  # 点击的地方小红圆点显示
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,  # 点击的地方显示坐标数字 参数1图片，参数2添加的文字，参数3左上角坐标，参数4字体，参数5字体粗细
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)  # 显示图片

    cv2.namedWindow("image")  # 定义图片窗口
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)  # 回调函数，参数1窗口的名字，参数2鼠标响应函数
    cv2.imshow("image", img)  # 显示图片
    cv2.waitKey(0)

    c = []  # 用于存放所有坐标
    for i in range(0, len(a)):
        print(a[i], b[i])  # 打印坐标
        c.append([a[i], b[i]])
        print(c)

    if len(c) != 4:
        predict_info_show = '请重新运行，并只能点击4个坐标……'
        # exit(-1)
    elif len(c) == 4:
        # cv2.destroyWindow('image')#关闭该窗口
        cv2.destroyAllWindows()
        predict_info_show = '输入成功，等待处理中'
    return c

def displayImg_out(mainwin, img):
    img = mainwin.padding(img)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
    img_out = QPixmap(img_out)
    # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
    img_out = mainwin.resizeImg(img_out)
    mainwin.label_out.setPixmap(img_out)

def displayImg_in(mainwin, img):
    img = mainwin.padding(img)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
    img_out = QPixmap(img_out)
    # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
    img_out = mainwin.resizeImg(img_out)
    mainwin.label_in.setPixmap(img_out)

def run(source, img, c1, c2, mainwin):
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    h, w = img.shape[:2]   #得到图片帧的宽和高
    mask_image_temp = np.zeros((h, w), dtype=np.uint8)

    # 初始化2个撞线polygon
    list_pts_blue = [c1[0], c1[1], c1[2], c1[3]]  # 蓝色区域，进入检测
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]


    mask_image_temp = np.zeros((h, w), dtype=np.uint8)
    # 填充第二个polygon
    list_pts_red = [c2[0], c2[1], c2[2], c2[3]]  # 红色区域，离开检测
    ndarray_pts_red = np.array(list_pts_red, np.int32)
    polygon_red_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=2)
    polygon_red_value_2 = polygon_red_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_red = polygon_blue_value_1 + polygon_red_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_red = cv2.resize(polygon_mask_blue_and_red, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 红 色盘
    red_color_plate = [193, 182, 255]
    # 红 polygon图片
    red_image = np.array(polygon_red_value_2 * red_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + red_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与红色polygon重叠
    list_overlapping_red_polygon = []


    # 进入数量
    enter_count = 0
    # 离开数量
    leave_count = 0


    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(source)
    #创建txt结果保存文件
    if not os.path.exists('results_txt'):
        os.makedirs('results_txt')
    if '.' in str(source):
        file_name = 'results_txt/' + os.path.basename(source).split('.')[0] + '.txt'
    else:
        file_name = 'results_txt/camera.txt'

    last_time = 0
    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break
        if mainwin.stop:
            break

        displayImg_in(mainwin, im)

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes, num_person = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_red[y, x] == 1 and point_in_polygon(x*2, y*2, list_pts_blue):
                    # 如果撞 蓝polygon，将id加入到blue list中
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 红polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_red_polygon:
                        # 进入+1
                        enter_count += 1

                        print(f'id: {track_id} | 进入 | 目前进入总数总数: {enter_count} | 蓝色区域id列表: {list_overlapping_blue_polygon} | 红色区域id列表: {list_overlapping_red_polygon}')

                        # 删除 红polygon list 中的此id
                        list_overlapping_red_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_red[y, x] == 2 and point_in_polygon(x*2, y*2, list_pts_red):
                    # 如果撞 红polygon，将id加入到red list中
                    if track_id not in list_overlapping_red_polygon:
                        list_overlapping_red_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 离开方向
                    if track_id in list_overlapping_blue_polygon:
                        # 离开+1
                        leave_count += 1

                        print(f'id: {track_id} | 离开 | 目前离开总数: {leave_count} | 蓝色区域id列表: {list_overlapping_blue_polygon} | 红色区域id列表: {list_overlapping_red_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_red_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_red_polygon:
                        list_overlapping_red_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_red_polygon.clear()
            pass
        pass

        # text_draw = 'DOWN: ' + str(leave_count) + \
        #             ' , UP: ' + str(enter_count)
        # output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
        #                                  org=draw_text_postion,
        #                                  fontFace=font_draw_number,
        #                                  fontScale=1, color=(255, 255, 255), thickness=2)
        mainwin.listWidget_out.clear()
        mainwin.listWidget_out.addItem('进入人数:  ' + str(int(enter_count)))
        mainwin.listWidget_out.addItem('离开人数:  ' + str(int(leave_count)))
        mainwin.listWidget_out.addItem('总人数:   ' + str(int(num_person)))
        mainwin.listWidget_out.addItem('内部人数:   ' + str(int(enter_count) - int(leave_count)))
        mainwin.listWidget_out.addItem('人数超限: ' + str(int(num_person) > mainwin.num_thres))
        if int(time.time())%10 == 0 and int(time.time()) != last_time:
            # print(time.time())
            info = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '\t'
            info += '进入人数:' + str(int(enter_count)) + '\t'
            info += '离开人数:' + str(int(leave_count)) + '\t'
            info += '总人数:' + str(int(num_person)) + '\n'
            with open(file_name,"a") as f:
                f.write(info)  # 自带文件关闭功能，不需要再写f.close()
        last_time = int(time.time())
        # cv2.imshow('demo', output_image_frame)
        # cv2.waitKey(1)
        output_image_frame=cv2.resize(output_image_frame, (w,h))
        displayImg_out(mainwin, output_image_frame)
        pass
    pass

    capture.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     root = 'video/test.mp4'  #输入需要处理的视频路径
#     # saveroot = "./video/out001.avi" #视频保存的地址
#     frame = getframe(root)   #得到视频的一张图片保存，用于后续选择坐标
#     c1 = zuobiao(frame) #用于手动在图像上确定检测范围,返回值为4个坐标点
#     c2 = zuobiao(frame) #用于手动在图像上确定检测范围,返回值为4个坐标点
#     run(frame,c1,c2) #运行检测跟踪函数
