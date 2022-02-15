import cv2
import csv
import os
import csv
import utils
from time import sleep, time
from face_mesh import FaceMesh
import json

marker_path = 'dataset/markers/'
video_path = 'dataset/YawDD/train/'
dirFile = os.listdir(marker_path)

def main():
    fileClean()
    ''' Create object '''
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    ''' os '''
    fileList = []
    train_Y = []
    for f in dirFile:
        if f[-4:] == '.csv':
            fileList.append(f[:-4])
    ttnum = 0
    for i, f in enumerate(fileList):
        marker_file_path = f'{marker_path}{f}.csv'
        video_file_path = f'{video_path}{f}.avi'
        train_y = read_csv(marker_file_path)
        print(f'{f}, len = {len(train_y)}')
        fileAppendTextLine(json.dumps({
            'file' : f,
            'size' : len(train_y),
            'fps' : 30,
            'marker_path' : marker_file_path,
            'video_path' : video_file_path
        }))
        ''' CaptureInput '''
        # with FaceMesh(1, 0.7, 0.7) as face_mesh:
        face_mesh = FaceMesh(1, 0.7, 0.7)
        cap = cv2.VideoCapture(video_file_path)
        last_rotation = 0, 0, 0
        for times in range(len(train_y)):
            ttnum += 1
            # print(f'times = {times}')
            display_fps = cvFpsCalc.get()
            ret, frame = cap.read()
            # print(ret)
            # print(f'{i + 1:3}.[{time():.2f}] {times + 1}/{len(train_y)}, {ttnum}')
            face_landmarks = face_mesh(frame)
            mouth_ear = 0
            rotation = 0, 0, 0
            for face_landmark in face_landmarks:
                ''' mouth detection '''
                mouth_ear = calc_mouth_ear(face_landmark)
                # face_bbox = face_mesh.calc_face_bbox(face_landmark)
                rotation = face_mesh.get_rotation(face_landmark)
                frame = draw_mouth_edge(frame.copy(), face_landmark)
            ''' display '''
            frame = draw_msg(frame.copy(), (
                f'FPS: {display_fps:.2f}',
                f'Sec: {times / 30:.2f}',
                f'Frame: {times + 1}/{len(train_y)}',
                f'Y_train:',
                f'  Level: {train_y[times]}',
                f'X_train:',
                f'  mouth_ear: {mouth_ear:.2f}',
                f'  roll: {rotation[0] - last_rotation[0]:.2f}',
                f'  yaw: {rotation[1] - last_rotation[1]:.2f}',
                f'  pitch: {rotation[2] - last_rotation[2]:.2f}'
            ))
            fileAppendTextLine(json.dumps({
                'index' : times + 1,
                'level' : train_y[times],
                'mouth_ear' : mouth_ear,
                'roll' : rotation[0] - last_rotation[0],
                'yaw' : rotation[1] - last_rotation[1],
                'pitch' : rotation[2] - last_rotation[2]
            }))
            last_rotation = rotation
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27: # ESC
                print('exit()')
                exit()
        cap.release()
        # cv2.destroyAllWindows()

def fileClean():
    with open('train_data.txt', 'w', encoding='utf-8') as file:
        file.write('')

def fileAppendTextLine(msg):
    with open('train_data.txt', 'a', encoding='utf-8') as file:
        file.write(f'{msg}\n')

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        for row in rows[1:]:
            data.append(row[0])
    return data

def draw_msg(image, str_arr):
	for i, s in enumerate(str_arr):
		cv2.putText(image, s, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	return image

def calc_mouth_ear(landmarks):
	p1 = landmarks[61]
	p2 = landmarks[81]
	p3 = landmarks[311]
	p4 = landmarks[291]
	p5 = landmarks[402]
	p6 = landmarks[178]
	ear = (utils.get_distance(p2, p6) + utils.get_distance(p3, p5))\
			/ (2 * utils.get_distance(p1, p4))
	return ear

def draw_mouth_edge(image, landmarks):
	pos = (61, 81, 311, 291, 402, 178)
	for p in pos:
		cv2.circle(image, landmarks[p][:2], 2, (0, 0, 255), -1)
	return image

if __name__ == '__main__':
    main()