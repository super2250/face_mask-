import cv2
import tensorflow as tf
import numpy as np
from scipy.special import softmax
import dlib


class mask():
    def __init__(self):
        self.model = tf.keras.models.load_model(r'D:\PYTHON\shizhanxiangmu\CNN\cnn人脸口罩佩戴识别\mark_model')
        self.face_mode = dlib.get_frontal_face_detector()

    def face_det(self,img):
        face_id = self.face_mode(img,1)
        for face in face_id:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(img,(l,t),(r,b),(200,0,190),5)
        return img

    def blob_pre(self,img):
        #人脸检测器
        img = self.face_det(img)
        # 转为blob
        img_blob = cv2.dnn.blobFromImage(img, 1, (100, 100), (104, 177, 123), swapRB=True)
        # 压缩维度、转置
        img_squeeze = np.squeeze(img_blob).T
        # 旋转
        img_a = cv2.rotate(img_squeeze, cv2.ROTATE_90_CLOCKWISE)
        # 镜像翻转
        img = cv2.flip(img_a, 1)
        # 把负数变为0
        img_blob = np.maximum(img, 0) / img.max()
        #变成4维
        img_blob = img_blob.reshape(1,100,100,3)
        #模型预测
        result = self.model.predict(img_blob)
        results = softmax(result[0])
        max_index = results.argmax()  #最大值索引
        max_val = results[max_index] #索引最大值
        line =['mask','no_mask']
        txt = f'{line[max_index]}:{round(max_val*100,2)}'
        return txt


    def rec(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret ,frame = cap.read()
            frame = cv2.flip(frame,1)

            txt= self.blob_pre(frame)
            cv2.putText(frame,txt, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv2.imshow('img',frame)
            if ord('q') == cv2.waitKey(1):
                break
        cap.release()
        cv2.destroyAllWindows()

s=mask()
s.rec()

