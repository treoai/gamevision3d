import sys
import cv2
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from processor import (  # updated to your filename
    Mode,
    run_pitch_detection,
    run_player_detection,
    run_ball_detection,
    run_player_tracking,
    run_team_classification
)

MODE_RUNNERS = {
    Mode.PITCH_DETECTION: run_pitch_detection,
    Mode.PLAYER_DETECTION: run_player_detection,
    Mode.BALL_DETECTION: run_ball_detection,
    Mode.PLAYER_TRACKING: run_player_tracking,
    Mode.TEAM_CLASSIFICATION: run_team_classification
}


class SoccerAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soccer Analyzer")
        self.setGeometry(100, 100, 960, 720)

        self.video_path = None
        self.device = "cuda"  # or "cpu"
        self.processing = False
        self.stop_event = threading.Event()

        self.init_ui()

    def init_ui(self):
        self.video_label = QLabel("Load a video to start")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([mode.name for mode in Mode])

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_processing)

        hbox = QHBoxLayout()
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.mode_combo)
        hbox.addWidget(self.start_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File")
        if file_name:
            self.video_path = file_name
            self.video_label.setText(f"Loaded: {file_name}")

    def toggle_processing(self):
        if not self.processing:
            if not self.video_path:
                self.video_label.setText("Please load a video first.")
                return

            self.processing = True
            self.stop_event.clear()
            self.start_button.setText("Stop")

            mode_str = self.mode_combo.currentText()
            mode = Mode[mode_str]
            runner = MODE_RUNNERS[mode]

            self.processing_thread = threading.Thread(
                target=self.run_analysis, args=(runner,), daemon=True)
            self.processing_thread.start()
        else:
            self.stop_event.set()
            self.processing = False
            self.start_button.setText("Start")
            self.video_label.setText("Processing stopped. Load a new video or start again.")

    def run_analysis(self, runner_func):
        for frame in runner_func(self.video_path, self.device):
            if self.stop_event.is_set():
                break
            if frame is not None:
                self.display_frame(frame)
        self.reset_after_processing()

    def display_frame(self, frame: np.ndarray):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        ))

    def reset_after_processing(self):
        self.processing = False
        self.start_button.setText("Start")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SoccerAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())
