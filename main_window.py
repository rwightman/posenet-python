"""Shows main page, connect front and backend"""
import sys

from IPython.external.qt_for_kernel import QtGui
from PyQt5 import QtWidgets, uic

import count_exercises
from forward_bends_knee import forward_bends_knee
from hands_up import hands_up
from head_ex import head_ex
from lift_leg import lift_leg

QT_FILE = "gui/med_rehab.ui"
UI_WINDOW, _ = uic.loadUiType(QT_FILE)


class MainWindow(QtWidgets.QMainWindow, UI_WINDOW):
    """Shows main window"""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        UI_WINDOW.__init__(self)

        self.setupUi(self)
        self.setWindowTitle("Medical Rehab App")

        self.home()

    def home(self):
        btn = QtGui.QPushButton("Quit", self)
        btn.clicked.connect(self.close_application)
        btn.resize(btn.minimumSizeHint())
        btn.move(110, 400)

        btn = QtGui.QPushButton("Go to exercise", self)
        btn.clicked.connect(self.go_to_exercise)
        btn.resize(btn.minimumSizeHint())
        btn.move(400, 400)
        self.cb = QtGui.QComboBox(self)
        self.cb.addItems(["Squats", "Lifting right hand", "Lifting left hand", "Lifting left leg", "Lifting right leg",
                          "Bends over", "Head's side bends"])
        self.cb.resize(self.cb.minimumSizeHint())
        self.cb.move(110, 250)

        self.show()

    def exercise_changed(self):
        pass

    def close_application(self):
        choice = QtGui.QMessageBox.question(self, 'Confirm exit',
                                            "Are you sure you want to exit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            print("Closing rehab app")
            sys.exit()
        else:
            pass

    def go_to_exercise(self):

        amount = WINDOW.amountEdit.text()
        print(amount)
        if self.cb.currentText() == "Squats":
            count_exercises.main(amount, 'squart')
        elif self.cb.currentText() == "Lifting right hand":
            hands_up("right")
        elif self.cb.currentText() == "Lifting left hand":
            hands_up("left")
        elif self.cb.currentText() == "Lifting right leg":
            lift_leg("right")
        elif self.cb.currentText() == "Lifting left leg":
            lift_leg("left")
        elif self.cb.currentText() == "Bends over":
            forward_bends_knee()
        elif self.cb.currentText() == "Head's side bends":
            head_ex()


if __name__ == '__main__':
    APP = QtWidgets.QApplication(sys.argv)
    WINDOW = MainWindow()
    WINDOW.show()
    APP.exec_()
