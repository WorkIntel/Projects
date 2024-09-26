
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the Title of the window
        self.setWindowTitle('Mouse and Keyboard Event Demo')
        # Set the position and size of the window
        self.setGeometry(100, 100, 400, 300)

    # This method checks if the left mouse button was pressed on the widget
    # and prints the position of the click.
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            print("Left mouse button pressed at:", event.position())

# Initialize the QApplication
app = QApplication([])
window = MainWindow()
window.show()
app.exec()  # Start the event loop