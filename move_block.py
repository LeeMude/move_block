import cv2
import cv2 as cv
import numpy as np
import time
import mediapipe as mp

#手部函数检测
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#绘制关键点和连接线函数
mpDraw = mp.solutions.drawing_utils
#handLmsStyle和handConStyle分别是关键点和连接线的特征，包括颜色和关键点（连接线）的宽度。
#如果画面中有手，就可以通过如下函数将关键点和连接线表示出来
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))



#调用摄像头
cap = cv2.VideoCapture(0)


# 获取画面宽度、高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

x=100
y=100
w=100
h=100
d = 0
while True:
    #返回frame图片
    rec,frame = cap.read()

    #镜像
    frame = cv.flip(frame,1)

    frame.flags.writeable = False
    # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 返回结果
    results = hands.process(frame)

    frame.flags.writeable = True
    #转换回来
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #如果识别到手
    if results.multi_hand_landmarks:
        #遍历每一只手
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                frame,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS,
                handLmsStyle,
                handConStyle)

            # 更新存储手指的坐标
            f_x = []
            f_y = []

            #记录手指的坐标
            for landmark in hand_landmarks.landmark:
                f_x.append(landmark.x)
                f_y.append(landmark.y)


        # 判断食指和中指是否合并
        finger_x1, finger_y1 = int(f_x[8] * width), int(f_y[8] * height)
        finger_x2, finger_y2 = int(f_x[12] * width), int(f_y[12] * height)
        #print((finger_x1, finger_y1))
        d = np.sqrt((finger_x1 - finger_x2) ** 2 + (finger_y1 - finger_y2) ** 2)
        #print(d)

        if d <= 30:
            move = True
        else:
            move = False

        # 更新方块的坐标
        if move == True:
            x = int(finger_x1 - w / 2)
            y = int(finger_y1 - h / 2)

        # 限制方块跑出边界
        if x < 0:
            x = 0
        elif x > width - w:
            x = width - w
        if y < 0:
            y = 0
        elif y > height - h:
            y = height - h








    frame_copy = frame.copy()
    #画矩形
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-1)#最后一个参数是线宽，为负数时表示为实心
    #使方框变透明
    #透明度
    k = 0.5
    frame = cv.addWeighted(frame_copy,1-k,frame,k,0)
    #画画面
    cv.imshow('frame',frame)

    #退出条件
    if cv.waitKey(1) & 0xff == ord(' '):#cv.waitKey(1)当按下按键时返回相应的ASCII值，否则返还-1  ord('q')为q的ASCII值
        break
cap.release()
cv.destroyAllWindows()
