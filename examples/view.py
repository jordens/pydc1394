from __future__ import (print_function, unicode_literals, division,
        absolute_import)

import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from pydc1394 import Camera


class CameraPlot:
    def __init__(self, camera):
        self.camera = camera
        self.init_win()
        self.init_camera()
    
    def init_win(self):
        self.win = QtGui.QMainWindow()
        self.win.show()
        self.win.resize(600, 400)
        self.win.setWindowTitle("pydc1394 + pyqtgraph")
        self.img = pg.ImageView()
        self.win.setCentralWidget(self.img)

    def init_camera(self):
        print("Vendor:", self.camera.vendor)
        print("Model:", self.camera.model)
        print("GUID:", self.camera.guid)
        print("Mode:", self.camera.mode)
        print("Framerate: ", self.camera.rate)
        print("Available modes", [mode.name for mode in self.camera.modes])
        print("Available features", self.camera.features)
        #modes = self.camera.modes
        #self.camera.mode = modes[0]
       
    def start_camera(self):
        self.camera.start_capture()
        self.camera.start_video()

    def process_images(self):
        QtCore.QTimer.singleShot(50, self.process_images)
        frame = None
        while True:
            frame_ = self.camera.dequeue(poll=True)
            if frame_ is not None:
                if frame is not None:
                    frame.enqueue()
                frame = frame_
            else:
                break
        if frame is None:
            return
        im = frame.copy().T
        frame.enqueue()
        self.img.setImage(im, autoRange=False, autoLevels=False,
            autoHistogramRange=False)

    def stop_camera(self):
        self.camera.stop_video()
        self.camera.stop_capture()
        
    def deinit_camera(self):
        pass

    
if __name__ == "__main__":
    app = QtGui.QApplication([])
    cam = CameraPlot(Camera())
    try:
        cam.start_camera()
        time.sleep(.5)
        cam.process_images()
        cam.img.autoRange()
        cam.img.autoLevels()
        QtGui.QApplication.instance().exec_()
    finally:
        cam.stop_camera()
        cam.deinit_camera()
