# encoding: utf-8

from ctypes.wintypes import LANGID
import os
import time
import cv2
import dlib
from matplotlib import pyplot as plt

from common_utils.check_utils import check_model_path
from dlib_face_detection import CnnFaceDetector


class FaceAlignment:
    def __init__(self, num_landmarks=68) -> None:
        self.num_landmarks = num_landmarks
        if self.num_landmarks == 68:
            self.model_path = (
                "models/face_recognition_models/shape_predictor_68_face_landmarks.dat"
            )
        elif self.num_landmarks == 5:
            self.model_path = (
                "models/face_recognition_models/shape_predictor_5_face_landmarks.dat"
            )
        else:
            raise ValueError(
                "Invalid number of landmarks: {}, num_landmarks can only be set as 68 or 5".format(
                    self.num_landmarks
                )
            )
        check_model_path(self.model_path)
        self.predictor = dlib.shape_predictor(self.model_path)

    def load_image(self, img_path):
        """载入图像"""
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image file not found: {}".format(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Failed to load image: {}".format(img_path))
        return img

    def __call__(self, image, face_rect):
        """检测人脸"""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.predictor(image, face_rect)

    def demo(self, img_path):
        """展示示例"""
        image = self.load_image(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 人脸检测
        detector = CnnFaceDetector()
        face_rects = detector(gray)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        time_stat = 0
        for face_rect in face_rects:
            # 人脸对齐的轮廓点
            start_time = time.time()
            landmarks = self(gray, face_rect)
            end_time = time.time()
            time_stat += end_time - start_time
            # 可视化
            print(landmarks.parts())
            for landmark in landmarks.parts():
                plt.plot(landmark.x, landmark.y, "ro", markersize=1)
        print("Average cost time(s): {:.4f}".format(time_stat / len(face_rects)))
        plt.savefig("output/dlib_face_alignment.png", bbox_inches="tight", dpi=300)
        plt.show()


def main():
    img_path = "data/face-samples/two_people.jpeg"
    face_alignment = FaceAlignment(num_landmarks=68)
    face_alignment.demo(img_path)


if __name__ == "__main__":
    main()
