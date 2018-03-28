import cv2
import threading
import time


class VideoCapture:
    def __init__(self, path):
        self.path = path
        self.capture = cv2.VideoCapture(self.path)
        self.running = False
        self.__get_next = False

        self.lock = threading.Lock()
        self.frame_rv, self.frame = self.capture.read()
        self.thread = None
        assert self.frame_rv

    def set(self, property, value):
        return self.capture.set(property, value)

    def get(self, property):
        return self.capture.get(property)

    def start(self):
        print('Starting capture!')
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if self.__get_next:
                rv, frame = self.capture.read()
                with self.lock:
                    self.frame = frame
                    self.frame_rv = rv
                    self.__get_next = False
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            frame = self.frame.copy()
            rv = self.frame_rv
            self.__get_next = rv
        return rv, frame

    def stop(self):
        self.running = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()