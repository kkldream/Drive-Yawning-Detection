import cv2
import numpy as np
import utils
from face_mesh import FaceMesh

def main():
	cap = utils.CaptureInput(0, 640, 480, 30)
	cap.setFlip = True
	face_mesh = FaceMesh(1, 0.7, 0.7)
	cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
	while True:
		display_fps = cvFpsCalc.get()
		ret, frame = cap.read()
		face_results = face_mesh(frame)
		for face_result in face_results:
			''' face bbox '''
			face_bbox = face_mesh.calc_face_bbox(face_result)
			face_mid = face_mesh.calc_face_mid(face_result)
			roll, yaw, pitch = face_mesh.get_rotation(face_result)
			all_face_bbox = [
				face_bbox[0],
				(face_bbox[0][0], face_bbox[1][1]),
				face_bbox[1],
				(face_bbox[1][0], face_bbox[0][1])
			]
			rotate_face_bbox = [utils.rotate_xy(all_face_bbox[i], face_mid, roll * -1) for i in range(4)]
			for i in range(-1, 3):
				cv2.line(frame, rotate_face_bbox[i], rotate_face_bbox[i+1], (255, 0, 0))
			''' face 486 point '''
			for face_landmarks in face_result:
				pos = face_landmarks[:2]
				depth = face_landmarks[2]
				if depth >= 0:
					cv2.circle(frame, pos, 1, (0, 0, int(255 * depth * 50)), -1)
				else:
					cv2.circle(frame, pos, 1, (int(255 * depth * -1 * 50), 0, 0), -1)
		''' display '''
		frame = draw_msg(frame.copy(), (
			f'FPS: {display_fps:.2f}',
			f'Roll: {roll}',
			f'Yaw: {yaw}',
			f'Pitch: {pitch}'
		))
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == 27: # ESC
			break
	cap.release()
	cv2.destroyAllWindows()
	return

def draw_msg(image, str_arr):
	for i, s in enumerate(str_arr):
		cv2.putText(image, s, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	return image

if __name__ == '__main__':
	main()
