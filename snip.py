import os
import sys
import cv2

from PyQt5.QtGui import QPainter, QPainterPath, QPixmap, QColor
from PyQt5.QtCore import QRect, Qt, QSize
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton, QWidget

from png2fen import evaluate


def grabScreenshot():
    desktopPixmap = QPixmap(QApplication.desktop().geometry().size())
    p = QPainter(desktopPixmap)
    for screen in QApplication.screens():
        p.drawPixmap(screen.geometry().topLeft(), screen.grabWindow(0))

    return desktopPixmap


class SelectorWidget(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.desktop().geometry())

        self.desktopPixmap = grabScreenshot()
        self.selectedPixmap = QPixmap()
        self.selectedRect = QRect()

    def mousePressEvent(self, event):
        self.selectedRect.setTopLeft(event.globalPos())

    def mouseMoveEvent(self, event):
        self.selectedRect.setBottomRight(event.globalPos())
        self.update()

    def mouseReleaseEvent(self, event):
        self.selectedPixmap = self.desktopPixmap.copy(self.selectedRect.normalized())
        self.accept()
        self.selectedPixmap.save('temp', 'png')
        fen = evaluate('temp')
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(fen, mode=cb.Clipboard)
        os.remove('temp')

    def paintEvent(self, event):
        paint = QPainter(self)
        paint.drawPixmap(0, 0, self.desktopPixmap)

        path = QPainterPath()
        path.addRect(*self.selectedRect.getRect())
        paint.fillPath(path, QColor.fromRgb(255, 255, 255, 200))

        paint.setPen(Qt.red)
        paint.drawRect(self.selectedRect)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Snipping tool')

        button = QPushButton('Snip!')
        button.clicked.connect(self.snipped)

        self.setCentralWidget(button)

    def snipped(self, s):
        w = SelectorWidget()
        w.show()
        w.exec_()


app = QApplication(sys.argv)
window = MainWindow()
window.show()

sys.exit(app.exec_())
