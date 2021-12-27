'''
import utils
cap = CaptureInput(0, 640, 480, 30)
'''
import cv2

class CaptureInput(cv2.VideoCapture):
    flip = 1
    isVideo = False
    
    def __init__(self, var, width=640, height=480, fps=30, info=True) -> None:
        super().__init__(var)
        if type(var) == str:
            self.isVideo = True
        self.setFps(fps)
        self.setSize(width, height)
        if info is True:
            self.info()

    def info(self):
        print(f'Resolution = {self.width}x{self.height}, FPS = {self.fps}')
    
    def setSize(self, width, height):
        self.width = width
        self.height = height
        self.shape = width, height
        self.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def setFps(self, fps):
        self.fps = fps
        self.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ret, frame = super().read()
        if self.isVideo:
            frame = cv2.resize(frame, self.shape)
        frame = cv2.flip(frame, self.flip)
        return ret, frame

    def setFlip(self, var):
        self.flip = var

if __name__ == '__main__':
    cap = CaptureInput(0, 640, 480, 30)
    cap.setDisplayFps(True)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()