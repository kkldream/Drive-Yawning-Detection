import cv2
import numpy as np

class KalmanFilter:
	def __init__(self, num=1, noise=0.1) -> None:
		self.num = num
		measurementMatrix = np.zeros((self.num, self.num * 2), np.float32)
		for i in range(self.num): measurementMatrix[i][i] = 1
		processNoiseCov = np.zeros((self.num * 2, self.num * 2), np.float32)
		for i in range(self.num * 2): processNoiseCov[i][i] = noise
		self._kalman = cv2.KalmanFilter(self.num * 2, self.num)
		self._kalman.measurementMatrix = measurementMatrix
		self._kalman.processNoiseCov = processNoiseCov
		
	def __call__(self, measurement):
		measurement = np.array(measurement, np.float32)
		self._kalman.correct(measurement)
		prediction = self._kalman.predict()
		return prediction[:self.num, 0]