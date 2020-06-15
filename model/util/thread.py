# -*- coding: UTF-8 -*-

import threading
import time

from ..conf import conf
from .GUI import Ui_Form
from PySide2 import QtGui


class QTThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.GUI = Ui_Form()
        self.name = name
        self.is_loop = True
        self.process_ready = False
        self.image = None
        self.GUI.show()

    def run(self):
        while self.is_loop and self.GUI.close_flag:
            if self.process_ready:
                self.GUI.show_video.setPixmap(QtGui.QPixmap.fromImage(self.image))
                self.process_ready = False
            else:
                time.sleep(0.01)

    def stop(self):
        self.is_loop = False

    def info(self, msg):
        self.GUI.show_message.append(msg)

    def warn(self, msg):
        self.GUI.show_message2.append(msg)

    def set_image(self, image):
        self.image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
