import cv2
import numpy as np
import utils
import time
from face_mesh import FaceMesh
import json
from keras.models import load_model
my_model = load_model('saved_models/model-81_epoch-0.00209_loss.hdf5')
def main():
	level = 0
	TestList = np.zeros((300, 4))
	''' CaptureInput '''
	# cap = utils.CaptureInput(0, 640, 480, 30)
	cap = utils.CaptureInput('dataset/YawDD/test/4-MaleNoGlasses.avi')
	# cap = utils.CaptureInput('blink_test.mp4', 960, 540)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	last_rotation = 0, 0, 0
	''' Start Loop'''
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		rotation = 0, 0, 0
		for face_result in face_results:
			''' mouth detection '''
			mouth_ear = calc_mouth_ear(face_result)
			rotation = face_mesh.get_rotation(face_result)
			frame = draw_mouth_edge(frame.copy(), face_result)
			Test = np.array([
				map(mouth_ear, 0, 1, 0, 1),
				map(rotation[0] - last_rotation[0], -50, 50, -1, 1),
				map(rotation[1] - last_rotation[1], -50, 50, -1, 1),
				map(rotation[2] - last_rotation[2], -50, 50, -1, 1)
			])
			cv2.putText(frame, f'{int(mouth_ear*100)}%', face_result[81][:2],
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			TestList[:-1] = TestList[1:]
			TestList[-1] = Test
			predicted = my_model.predict(np.array([TestList]))
			level = max(0, min(5, int(predicted[0,0] * 10)))
		last_rotation = rotation
		''' display '''
		cv2.putText(frame, "FPS:" + str(display_fps), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.putText(frame, "Level:" + str(level), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow('frame', frame)
		key = cv2.waitKey(1)
		if key == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()

def fileAppendTextLine(msg):
    with open('output2.txt', 'a', encoding='utf-8') as file:
        file.write(f'{msg}\n')

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

def map(var, in_min, in_max, out_min, out_max):
    var = float(var)
    if var >= in_max:
        return out_max
    if var <= in_min:
        return out_min
    return (var - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

if __name__ == '__main__':
	main()
