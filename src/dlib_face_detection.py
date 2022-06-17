# encoding: utf-8

import os
import time
import cv2
import dlib
import matplotlib.pyplot as plt


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        """检测人脸"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector(img, 1)

    def plot_rect(self, rect):
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        plt.gca().add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
        )

    def demo(self, img_path):
        """展示示例"""
        image = self.load_image(img_path)
        start_time = time.time()
        rects = self.detect(image)
        end_time = time.time()
        print(
            "Number of faces detected: {}, cost time(s): {:.4f}".format(
                len(rects), end_time - start_time
            )
        )
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for rect in rects:
            self.plot_rect(rect)
        plt.show()

    def load_image(self, img_path):
        """载入图像"""
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image file not found: {}".format(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Failed to load image: {}".format(img_path))
        return img


class CnnFaceDetector(FaceDetector):
    def __init__(self):
        self.detector = dlib.cnn_face_detection_model_v1(
            "models/face_recognition_models/mmod_human_face_detector.dat"
        )

    def detect(self, image):
        """检测人脸"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector(img, 1)

    def demo(self, img_path):
        """展示示例"""
        image = self.load_image(img_path)
        start_time = time.time()
        rects = self.detect(image)
        end_time = time.time()
        print(
            "Number of faces detected: {}, cost time(s): {:.4f}".format(
                len(rects), end_time - start_time
            )
        )
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for mmod_rect in rects:
            self.plot_rect(mmod_rect.rect)
        plt.show()


def main():
    img_path = "data/face-samples/two_people.jpeg"

    FaceDetector().demo(img_path)
    CnnFaceDetector().demo(img_path)


if __name__ == "__main__":
    main()
