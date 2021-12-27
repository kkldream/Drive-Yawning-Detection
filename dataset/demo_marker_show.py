import cv2
import csv

file_name = '1-FemaleNoGlasses-Yawning'
marker_path = f'dataset/markers/{file_name}.csv'
video_path = f'dataset/YawDD dataset/Mirror/Female_mirror/{file_name}.avi'

def main():
    data = read_csv(marker_path)
    cap = cv2.VideoCapture(video_path)
    frame_times = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = draw_msg(frame, (
            f'Frame: {frame_times}',
            f'Sec: {frame_times / 30:.2f}',
            f'Level: {data[frame_times]}'))
        cv2.imshow('frame', frame)
        frame_times += 1
        if cv2.waitKey(10) == 27: # ESC
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
		cv2.putText(image, s, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	return image

if __name__ == '__main__':
    main()