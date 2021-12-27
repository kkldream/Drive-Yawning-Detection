import cv2
import numpy as np
import matplotlib.pyplot as plt

class PltOpenCV:
	def __init__(self, xlim) -> None:
		self.arr = np.zeros(xlim)
	
	def __call__(self, y, ylim_start=-1, ylim_end=-1):
		self.append(y)
		plt_img = self.draw(ylim_start, ylim_end)
		return plt_img

	def append(self, y):
		self.arr[1:] = self.arr[:-1]
		self.arr[0] = y

	def draw(self, ylim_start=-1, ylim_end=-1):
		plt.plot(self.arr)
		if ylim_start != -1 and ylim_end != -1:
			plt.ylim(ylim_start, ylim_end)
		plt.savefig('plt_temp.png')
		plt.close()
		plt_img = cv2.imread('plt_temp.png')
		return plt_img

	def draw_arr(self, arr, ylim_start=-1, ylim_end=-1):
		plt.plot(arr)
		if ylim_start != -1 and ylim_end != -1:
			plt.ylim(ylim_start, ylim_end)
		plt.savefig('plt_temp.png')
		plt.close()
		plt_img = cv2.imread('plt_temp.png')
		return plt_img