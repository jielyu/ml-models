# encoding: utf-8

import sys
import unittest
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("src/common_utils")
from face_utils import FaceUtils


class TestTest(unittest.TestCase):
    """测试杂项"""

    def test_face_utils(self):
        """测试面部处理工具"""
        img_path = "data/idinvert-samples/1200px-Vladimir_Putin_(2020-02-20).jpg"
        img = cv2.imread(img_path)
        print(img.shape)
        face_utils = FaceUtils()
        aligned_face = face_utils.align(img, face_width=256)
        # 可视化
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("face aligned")
        plt.savefig("output/face_utils.png", bbox_inches="tight", dpi=300)
        # plt.show()


def main():
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    main()
