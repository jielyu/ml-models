# encoding: utf-8

import os
import time
import cv2
import dlib
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from common_utils.check_utils import check_model_path
from dlib_face_alignment import FaceAlignment
from dlib_face_detection import CnnFaceDetector


class FaceRecognition:
    def __init__(self) -> None:
        self.face_detector = CnnFaceDetector()
        self.face_alignment = FaceAlignment()
        face_encoder_model_path = (
            "models/face_recognition_models/dlib_face_recognition_resnet_model_v1.dat"
        )
        check_model_path(face_encoder_model_path)
        self.face_encoder = dlib.face_recognition_model_v1(face_encoder_model_path)

    def load_image(self, img_path):
        """载入图像"""
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image file not found: {}".format(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Failed to load image: {}".format(img_path))
        return img

    def predict_shape(self, gray, face_rects=None):
        """预测人脸轮廓点"""
        if face_rects is None:
            face_rects = self.face_detector(gray)
        landmarks = [self.face_alignment(gray, face_rect) for face_rect in face_rects]
        return landmarks

    def encode_faces(self, image, landmarks):
        """编码人脸"""
        return [
            np.array(self.face_encoder.compute_face_descriptor(image, landmark))
            for landmark in landmarks
        ]

    def compare_faces(self, known_face_encodings, face_encoding):
        """比较人脸"""
        if len(known_face_encodings) == 0 or face_encoding is None:
            raise ValueError("known_face_encodings or face_encoding is empty")
        return list(np.linalg.norm(known_face_encodings - face_encoding, axis=1))

    def bgr2rgb_gray(self, image):
        """格式化图像"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return gray, rgb

    def demo(self, img_path1, img_path2, DIST_THRESHOLD=0.6):
        """展示示例"""
        # 载入原始图像
        image1 = self.load_image(img_path1)
        image2 = self.load_image(img_path2)
        # 转换图像格式
        gray1, rgb1 = self.bgr2rgb_gray(image1)
        gray2, rgb2 = self.bgr2rgb_gray(image2)
        # 检测轮廓点
        landmarks1, landmarks2 = self.predict_shape(gray1), self.predict_shape(gray2)
        # 编码人脸
        start_time = time.time()
        encodings1 = self.encode_faces(rgb1, landmarks1)
        encodings2 = self.encode_faces(rgb2, landmarks2)
        end_time = time.time()
        print("Average Encode time: {}".format((end_time - start_time) / 2))
        # 比较人脸
        dist_list = [
            self.compare_faces(encodings1, encoding) for encoding in encodings2
        ]
        dist = np.array(dist_list)
        # 阈值判断
        hit_table = dist <= DIST_THRESHOLD
        print(hit_table)

        # 可视化
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(rgb1)
        for landmark in landmarks1:
            rect = landmark.rect
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            ax1.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor="green", linewidth=2)
            )
        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(rgb2)
        for landmark in landmarks2:
            rect = landmark.rect
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            ax2.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
            )
        # 绘制识别匹配关系
        num_face1, num_face2 = hit_table.shape[1], hit_table.shape[0]
        for i in range(num_face2):
            for j in range(num_face1):
                print(i, j, hit_table[i, j])
                if hit_table[i, j]:
                    print("test")
                    rect1 = landmarks1[j].rect
                    xy1 = [
                        rect1.left() + rect1.width() / 2,
                        rect1.top() + rect1.height() / 2,
                    ]
                    rect2 = landmarks2[i].rect
                    xy2 = [
                        rect2.left() + rect2.width() / 2,
                        rect2.top() + rect2.height() / 2,
                    ]
                    conn = ConnectionPatch(
                        xyA=xy1,
                        xyB=xy2,
                        coordsA="data",
                        coordsB="data",
                        axesA=ax1,
                        axesB=ax2,
                        color="r",
                    )
                    ax2.add_artist(conn)
                    ax1.plot(xy1[0], xy1[1], "ro")
                    ax2.plot(xy2[0], xy2[1], "ro")
        plt.savefig("output/dlib_face_recognision.png", bbox_inches="tight", dpi=300)
        plt.show()


def main():
    imgpath1 = "data/face-samples/two_people.jpeg"
    imgpath2 = "data/face-samples/obama.jpeg"
    fr = FaceRecognition()
    fr.demo(imgpath1, imgpath2)


if __name__ == "__main__":
    main()
