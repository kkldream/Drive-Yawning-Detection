import cv2
import utils
from face_mesh import FaceMesh

def main():
	''' CaptureInput '''
	cap = utils.CaptureInput(0, 640, 480, 10)
	cap.setFlip = True
	''' Create object '''
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	''' Start Loop'''
	while True:
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
			f'Mouth Ear: {mouth_ear:.2f}'
		))
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()

def draw_msg(image, str_arr):
	for i, s in enumerate(str_arr):
		cv2.putText(image, s, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
