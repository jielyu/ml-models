# encoding: utf-8

import sys

sys.path.append("src/")

import argparse
import os
import time
import cv2
import dlib
import matplotlib.pyplot as plt

from common_utils.check_utils import check_model_path


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, image):
        """检测人脸"""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector(image, 1)

    def plot_rect(self, rect):
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        plt.gca().add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
        )

    def plot_image_with_rect(self, image, rects):
        plt.clf()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for rect in rects:
            self.plot_rect(rect)
        plt.savefig("output/dlib_face_detection.png", bbox_inches="tight", dpi=300)
        plt.show()

    def demo(self, img_path):
        """展示示例"""
        image = self.load_image(img_path)
        start_time = time.time()
        rects = self(image)
        end_time = time.time()
        print(
            "Number of faces detected: {}, cost time(s): {:.4f}".format(
                len(rects), end_time - start_time
            )
        )
        self.plot_image_with_rect(image, rects)

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
        self.model_path = "models/face_recognition_models/mmod_human_face_detector.dat"
        check_model_path(self.model_path)
        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)

    def __call__(self, image):
        """检测人脸"""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mmod_rects = self.detector(image, 1)
        return [mm_rect.rect for mm_rect in mmod_rects]

    def demo(self, img_path):
        """展示示例"""
        image = self.load_image(img_path)
        start_time = time.time()
        rects = self(image)
        end_time = time.time()
        print(
            "Number of faces detected: {}, cost time(s): {:.4f}".format(
                len(rects), end_time - start_time
            )
        )
        self.plot_image_with_rect(image, rects)


def str2bool(v):
    """用于命令行解析bool类型的参数"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    img_path = "data/face-samples/two_people.jpeg"
    args = argparse.ArgumentParser()
    args.add_argument("--cnn", type=str2bool, default=False)
    args = args.parse_args()

    if args.cnn is False:
        FaceDetector().demo(img_path)
    else:
        CnnFaceDetector().demo(img_path)


if __name__ == "__main__":
    main()
