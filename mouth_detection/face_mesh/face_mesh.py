#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import utils

class FaceMesh(object):
    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ):
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(
        self,
        image,
    ):
        self.shape = image.shape
        # 推論
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self._face_mesh.process(image)

        # X,Y座標を相対座標から絶対座標に変換
        # [X座標, Y座標, Z座標, Visibility, Presence]のリストに変更
        face_mesh_results = []
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                face_mesh_results.append(
                    self._calc_landmarks(image, face_landmarks.landmark))
        return face_mesh_results

    def _calc_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_list = []
        for _, landmark in enumerate(landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_list.append((landmark_x, landmark_y, landmark.z,
                                  landmark.visibility, landmark.presence))
        return landmark_list

    def _calc_bounding_rect(self, landmarks):
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks):
            landmark_x = int(landmark[0])
            landmark_y = int(landmark[1])

            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def get_eye_landmarks(self, landmarks):
        # 目の輪郭の座標列を取得

        left_eye_landmarks = []
        right_eye_landmarks = []

        if len(landmarks) > 0:
            # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg
            # 左目
            left_eye_landmarks.append((landmarks[133][0], landmarks[133][1]))
            left_eye_landmarks.append((landmarks[173][0], landmarks[173][1]))
            left_eye_landmarks.append((landmarks[157][0], landmarks[157][1]))
            left_eye_landmarks.append((landmarks[158][0], landmarks[158][1]))
            left_eye_landmarks.append((landmarks[159][0], landmarks[159][1]))
            left_eye_landmarks.append((landmarks[160][0], landmarks[160][1]))
            left_eye_landmarks.append((landmarks[161][0], landmarks[161][1]))
            left_eye_landmarks.append((landmarks[246][0], landmarks[246][1]))
            left_eye_landmarks.append((landmarks[163][0], landmarks[163][1]))
            left_eye_landmarks.append((landmarks[144][0], landmarks[144][1]))
            left_eye_landmarks.append((landmarks[145][0], landmarks[145][1]))
            left_eye_landmarks.append((landmarks[153][0], landmarks[153][1]))
            left_eye_landmarks.append((landmarks[154][0], landmarks[154][1]))
            left_eye_landmarks.append((landmarks[155][0], landmarks[155][1]))

            # 右目
            right_eye_landmarks.append((landmarks[362][0], landmarks[362][1]))
            right_eye_landmarks.append((landmarks[398][0], landmarks[398][1]))
            right_eye_landmarks.append((landmarks[384][0], landmarks[384][1]))
            right_eye_landmarks.append((landmarks[385][0], landmarks[385][1]))
            right_eye_landmarks.append((landmarks[386][0], landmarks[386][1]))
            right_eye_landmarks.append((landmarks[387][0], landmarks[387][1]))
            right_eye_landmarks.append((landmarks[388][0], landmarks[388][1]))
            right_eye_landmarks.append((landmarks[466][0], landmarks[466][1]))
            right_eye_landmarks.append((landmarks[390][0], landmarks[390][1]))
            right_eye_landmarks.append((landmarks[373][0], landmarks[373][1]))
            right_eye_landmarks.append((landmarks[374][0], landmarks[374][1]))
            right_eye_landmarks.append((landmarks[380][0], landmarks[380][1]))
            right_eye_landmarks.append((landmarks[381][0], landmarks[381][1]))
            right_eye_landmarks.append((landmarks[382][0], landmarks[382][1]))

        return left_eye_landmarks, right_eye_landmarks
    
    def _get_mtcnn_landmarks(self, landmarks):
        mtcnn_landmarks = []
        if len(landmarks) > 0:
            mtcnn_landmarks.append((int((landmarks[159][0] + landmarks[145][0]) / 2),
                              int((landmarks[33][1] + landmarks[133][1]) / 2)))
            mtcnn_landmarks.append((int((landmarks[386][0] + landmarks[374][0]) / 2),
                              int((landmarks[263][1] + landmarks[362][1]) / 2)))
            mtcnn_landmarks.append((landmarks[1][0], landmarks[1][1]))
            mtcnn_landmarks.append((landmarks[76][0], landmarks[76][1]))
            mtcnn_landmarks.append((landmarks[306][0], landmarks[306][1]))
        return mtcnn_landmarks

    def calc_eye_bbox(self, landmarks):
        # 目に隣接するバウンディングボックスを取得

        left_eye_lm, right_eye_lm = self.get_eye_landmarks(landmarks)

        left_eye_bbox = self._calc_bounding_rect(left_eye_lm)
        right_eye_bbox = self._calc_bounding_rect(right_eye_lm)

        return left_eye_bbox, right_eye_bbox

    def calc_around_eye_bbox(self, landmarks, around_ratio=0.5):
        # 目の周囲のバウンディングボックスを取得

        left_eye_bbox, right_eye_bbox = self.calc_eye_bbox(landmarks)

        left_eye_bbox = self._calc_around_eye(left_eye_bbox, around_ratio)
        right_eye_bbox = self._calc_around_eye(right_eye_bbox, around_ratio)

        return left_eye_bbox, right_eye_bbox

    def _calc_around_eye(self, bbox, around_ratio=0.5):
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        cx = int(x + (w / 2))
        cy = int(y + (h / 2))
        square_length = max(w, h)
        x = int(cx - (square_length / 2))
        y = int(cy - (square_length / 2))
        w = square_length
        h = square_length

        around_ratio = 0.5
        x = int(x - (square_length * around_ratio))
        y = int(y - (square_length * around_ratio))
        w = int(square_length * (1 + (around_ratio * 2)))
        h = int(square_length * (1 + (around_ratio * 2)))

        return [x, y, x + w, y + h]

    def get_rotation(self, landmarks):
        face_top = landmarks[10]
        left_eye_left_corner = landmarks[226]
        right_eye_right_corner = landmarks[446]
        between_mouth_nose = landmarks[164]
        roll = int(math.atan((right_eye_right_corner[1] - left_eye_left_corner[1])\
                   / (right_eye_right_corner[0] - left_eye_left_corner[0]))\
                   * 180 / math.pi)
        yaw = int((right_eye_right_corner[2] - left_eye_left_corner[2]) * 400)
        pitch = int((face_top[2] - between_mouth_nose[2]) * 400)
        return roll, yaw, pitch

    def calc_face_bbox(self, landmarks):
        boxs = [0, 0, 0, 0]
        for i, landmark in enumerate(landmarks):
            pos = (landmark[0], landmark[1])
            if i == 0:
                boxs = [pos[0], pos[0], pos[1], pos[1]]
            else:
                boxs = [
                    min(pos[0], boxs[0]),
                    max(pos[0], boxs[1]),
                    min(pos[1], boxs[2]),
                    max(pos[1], boxs[3])
                ]
        return (boxs[0], boxs[2]), (boxs[1], boxs[3])
        face_top = landmarks[10]
        face_bottom = landmarks[152]
        face_left = landmarks[234]
        face_right = landmarks[454]
        face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
        left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
        right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
        left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
        right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
        top = max(left_top[1], right_top[1])
        bottom = max(left_bottom[1], right_bottom[1])
        left = max(left_top[0], left_bottom[0])
        right = max(right_top[0], right_bottom[0])
        return (left, top), (right, bottom)
        # roi_mid = int(max((right_bottom[0] + left_top[0]) / 2, (right_top[0] + left_bottom[0]) / 2)), int(max((left_bottom[1] + right_top[1]) / 2, right_bottom[1] + left_top[1]) / 2)
        # roi_wight = max(utils.get_distance(right_bottom, left_top), utils.get_distance(right_top, left_bottom))
        # roi_hidght = max(utils.get_distance(left_bottom, left_top), utils.get_distance(right_bottom, right_top))
        # roi_max_side = max(roi_wight, roi_hidght)
        
    def calc_face_mid(self, landmarks):
        face_top = landmarks[10]
        face_bottom = landmarks[152]
        face_left = landmarks[234]
        face_right = landmarks[454]
        face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
        return face_mid

    def get_align_roi(self, image, landmarks, roll=False):
        face_top = landmarks[10]
        face_bottom = landmarks[152]
        face_left = landmarks[234]
        face_right = landmarks[454]
        face_mid = (int((face_top[0] + face_bottom[0]) / 2), int((face_left[1] + face_right[1]) / 2))
        left_top = (int(face_left[0] - (face_mid[0] - face_top[0])), int(face_left[1] - (face_mid[1] - face_top[1])))
        right_top = (int(face_right[0] - (face_mid[0] - face_top[0])), int(face_right[1] - (face_mid[1] - face_top[1])))
        left_bottom = (int(face_left[0] - (face_mid[0] - face_bottom[0])), int(face_left[1] - (face_mid[1] - face_bottom[1])))
        right_bottom = (int(face_right[0] - (face_mid[0] - face_bottom[0])), int(face_right[1] - (face_mid[1] - face_bottom[1])))
        roi_mid = int(max((right_bottom[0] + left_top[0]) / 2, (right_top[0] + left_bottom[0]) / 2)), int(max((left_bottom[1] + right_top[1]) / 2, right_bottom[1] + left_top[1]) / 2)
        roi_wight = max(utils.get_distance(right_bottom, left_top), utils.get_distance(right_top, left_bottom))
        roi_hidght = max(utils.get_distance(left_bottom, left_top), utils.get_distance(right_bottom, right_top))
        roi_max_side = max(roi_wight, roi_hidght)
        if roll is False:
            roll = self.get_rotation(landmarks)[0]
        frame = utils.rotate_img(image, roll, roi_mid)
        boxs = (
            utils.rotate_xy(left_top, roi_mid, roll),
            utils.rotate_xy(right_top, roi_mid, roll),
            utils.rotate_xy(right_bottom, roi_mid, roll),
            utils.rotate_xy(left_bottom, roi_mid, roll)
        )
        max_side = max(
            boxs[2][1] - boxs[0][1],
            max(boxs[1][0], boxs[2][0]) - min(boxs[0][0], boxs[3][0])
        )
        face_roi = frame.copy()[
            max(0, int(roi_mid[1] - max_side / 2)):int(roi_mid[1] + max_side / 2),
            max(0, int(roi_mid[0] - max_side / 2)):int(roi_mid[0] + max_side / 2)
        ]
        return face_roi