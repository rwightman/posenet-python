"""Shows main page, connect front and backend"""
import sys
from PyQt5 import QtWidgets, uic

import count_exercises

QT_FILE = "gui/med_rehab.ui"
UI_WINDOW, _ = uic.loadUiType(QT_FILE)

class MainWindow(QtWidgets.QMainWindow, UI_WINDOW):
    """Shows main window"""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        UI_WINDOW.__init__(self)
        self.setupUi(self)
        self.squatsButton.pressed.connect(on_squats_button)
        self.armsButton.pressed.connect(on_arms_button)

def on_squats_button():
    """Run function to count squats"""

    amount = WINDOW.amountEdit.text()
    count_exercises.main(amount, 'squart')

def on_arms_button(): # TO DO implement arn function in count_exercises
    """Run function to count arms"""

    amount = WINDOW.amountEdit.text()
    #count_exercises.main(amount, 'arm')

if __name__ == '__main__':
    APP = QtWidgets.QApplication(sys.argv)
    WINDOW = MainWindow()
    WINDOW.show()
    APP.exec_()
