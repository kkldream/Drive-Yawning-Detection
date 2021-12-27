import cv2
import csv
import utils
from face_mesh import FaceMesh

file_name = '20-MaleGlasses-Talking&yawning'
marker_path = f'dataset/markers/{file_name}.csv'
video_path = f'dataset/YawDD/train/{file_name}.avi'

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(video_path)
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	data = read_csv(marker_path)
	frame_times = 0
	''' Start Loop'''
	while cap.isOpened():
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_landmarks = face_mesh(frame)
		for face_landmark in face_landmarks:
			''' mouth detection '''
			mouth_ear = calc_mouth_ear(face_landmark)
			frame = draw_mouth_edge(frame.copy(), face_landmark)
		''' display '''
		frame = draw_msg(frame.copy(), (
			f'FPS: {display_fps:.2f}',
			f'Sec: {frame_times / 30:.2f}',
			f'Frame: {frame_times}',
			f'Mouth Ear: {mouth_ear:.2f}',
			f'Level: {data[frame_times]}'
		))
		cv2.imshow('frame', frame)
		frame_times += 1
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()

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
		cv2.circle(image, landmarks[p][:2], 1, (0, 0, 255), -1)
	return image

if __name__ == '__main__':
	main()
