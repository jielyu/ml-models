# encoding: utf-8

from collections import OrderedDict
import cv2
import dlib
from matplotlib import pyplot as plt
import numpy as np


class FaceUtils:

    # For dlib’s 68-point facial landmark detector:
    FACIAL_LANDMARKS_68_IDXS = OrderedDict(
        [
            ("mouth", (48, 68)),
            ("inner_mouth", (60, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17)),
        ]
    )

    def __init__(self, is_cnn_detector=False) -> None:
        self.is_cnn_detector = is_cnn_detector
        if is_cnn_detector:
            self.face_detector = dlib.cnn_face_detection_model_v1(
                "models/face_recognition_models/mmod_human_face_detector.dat"
            )
        else:
            self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            "models/face_recognition_models/shape_predictor_68_face_landmarks.dat"
        )

    def detect_faces(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mmod_rects = self.face_detector(image, 1)
        if self.is_cnn_detector:
            return [mm_rect.rect for mm_rect in mmod_rects]
        return mmod_rects

    def get_face_landmarks(self, image, face_rect):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.shape_predictor(image, face_rect)
        # 转换坐标
        coords = np.zeros((shape.num_parts, 2), dtype=np.float32)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def align(self, image, face_width=256):
        """对齐人脸
        参考代码： https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/facealigner.py
        """
        desiredFaceWidth = face_width
        desiredLeftEye = (0.371, 0.480)
        desiredFaceHeight = desiredFaceWidth
        # 图像数据转换
        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        rects = self.detect_faces(gray)
        if len(rects) < 1:
            return None
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.get_face_landmarks(gray, rects[0])

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FaceUtils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FaceUtils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX**2) + (dY**2))
        desiredDist = desiredRightEyeX - desiredLeftEye[0]
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2),
        )

        # grab the rotation matrix for rotating and scaling the face
        # print(eyesCenter, angle, scale)
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


def main():
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
    # plt.savefig("output/face_utils.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
