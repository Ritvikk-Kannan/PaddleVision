import os
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Algorithm.match import Match
import cv2
from ini_api import API
import BallTrack
import numpy as np
import configparser


IMAGE_WIDTH = 1230
IMAGE_HEIGHT = 390
WAITING_TIME_BETWEEN_FRAMES_IN_MS=30


def resource_path(relative_path):
    base_path = os.getcwd()
    print(base_path)
    return os.path.join(base_path, relative_path)

def close_program():
    sys.exit()


class Ui(QtWidgets.QMainWindow):

    def __init__(self):
        self.timer_count = 0
        super(Ui, self).__init__()
        uic.loadUi(resource_path('pongping.ui'), self)

        self.uploadbutton.clicked.connect(self.upload_video)
        self.opencamerabutton.clicked.connect(self.open_camera)
        self.restartbutton.clicked.connect(self.restart)
        self.actionQuit.triggered.connect(close_program)

        self.uploadbutton.setEnabled(True)
        self.opencamerabutton.setEnabled(False)
        self.show()

    def update_table(self, res):
        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(str(res[0])))
        self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(str(res[1])))

    def append_event(self, value):
        rowcount = self.tableWidget_2.rowCount()
        if value:
            self.tableWidget_2.insertRow(rowcount)
            self.tableWidget_2.setItem(rowcount, 0, QtWidgets.QTableWidgetItem(value))
            self.tableWidget_2.setItem(rowcount, 1, QtWidgets.QTableWidgetItem(
                str(int(self.timer_count / WAITING_TIME_BETWEEN_FRAMES_IN_MS))))

    def upload_video(self):
        self.restart()
        filename = QFileDialog.getOpenFileName(None, 'Open File', os.getenv('HOME'))

        if filename[0]:
            # First, let the user map the court boundaries
            if self.map_court_boundaries(filename[0]):
                # If mapping was successful, run the analysis
                self.run(filename[0])
            else:
                # If mapping was cancelled, show a message
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Court mapping was cancelled.")
                msg.setWindowTitle("Mapping Cancelled")
                msg.exec_()

    def open_camera(self):
        pass

    def run(self, path):
        self.timer_count = 0

        # Create API Object
        api = API()
        # Capture the video from the path

        api.reload_config()

        cap = cv2.VideoCapture(path)
        _, frame = cap.read()

        # Get crop points from ini
        points = api.get_crop_points()

        # Crop the frame
        frame = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous = grayImage.copy()

        # holds a record of previous ball positions
        trajectories = []

        # Read the ini for the table and net boundaries
        boundaryFirstPlayer, boundarySecondPlayer, boundaryNet = api.get_stadium_points()

        axisTranslation = [(points[0][1], points[0][0]), (points[0][1], points[0][0]), (points[0][1], points[0][0]),
                           (points[0][1], points[0][0])]

        boundaryFirstPlayer = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(boundaryFirstPlayer, axisTranslation)]
        boundarySecondPlayer = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(boundarySecondPlayer, axisTranslation)]
        boundaryNet = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(boundaryNet, axisTranslation)]

        # Construct the match
        m = Match()
        m.defineTable(boundaryFirstPlayer, boundarySecondPlayer, boundaryNet)
        m.startMatch()

        while True:
            self.timer_count += 1
            # Read frame if end of file is reached break
            _, frame = cap.read()

            if frame is None:
                break

            # Crop frame
            frame = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

            # Call Ball Track pass (the frame cropped, previous cropped, trajectories, points) recieve ballCoord
            ballCoord, previous = BallTrack.get_ball_coordinates(frame, previous, trajectories, points)

            if ballCoord is not None:
                m.updateGame(ballCoord)

            # Draw Trajectory
            if len(trajectories) > 5:
                cv2.line(frame, trajectories[-1], trajectories[-2], (0, 0, 255), 5)
                cv2.line(frame, trajectories[-2], trajectories[-3], (0, 255, 0), 5)

            # Draw The Stadium
            Contour = np.array(boundaryFirstPlayer)
            cv2.drawContours(frame, [Contour], 0, (0,255,255), 2)
            Contour = np.array(boundarySecondPlayer)
            cv2.drawContours(frame, [Contour], 0, (0,255,255), 2)
            Contour = np.array(boundaryNet)
            cv2.drawContours(frame, [Contour], 0, (255,255,255), 2)

            # UpdateScore
            res = [(m.players[0]).getScore(), (m.players[1]).getScore()]

            self.update_table(res)
            self.append_event(m.printInfo())

            # Show the frame
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            self.label_4.setPixmap(QPixmap.fromImage(qImg).scaled(IMAGE_WIDTH, IMAGE_HEIGHT, Qt.KeepAspectRatio))
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            QApplication.processEvents()


    def restart(self):
        clear_list = ["", "", "", ""]
        self.update_table(clear_list)
        self.tableWidget_2.setRowCount(0)

    def map_court_boundaries(self, video_path):
        # Create API Object
        api = API()

        # Capture the first frame from the video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            return False

        # Get crop points from ini
        points = api.get_crop_points()

        # Crop the frame
        frame = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

        # Create a copy of the frame for drawing
        mapping_frame = frame.copy()

        # Create a window for mapping
        cv2.namedWindow('Map Court Boundaries')

        # Initialize points lists
        first_player_points = []
        second_player_points = []
        net_points = []

        # Create a callback function for mouse events
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # First 4 clicks for first player area
                if len(first_player_points) < 4:
                    first_player_points.append((x, y))
                    cv2.circle(mapping_frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(mapping_frame, f"P1-{len(first_player_points)}", (x + 5, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Next 4 clicks for second player area
                elif len(second_player_points) < 4:
                    second_player_points.append((x, y))
                    cv2.circle(mapping_frame, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(mapping_frame, f"P2-{len(second_player_points)}", (x + 5, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Last 4 clicks for net area
                elif len(net_points) < 4:
                    net_points.append((x, y))
                    cv2.circle(mapping_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(mapping_frame, f"N-{len(net_points)}", (x + 5, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw polygons as they are completed
                if len(first_player_points) == 4:
                    pts = np.array(first_player_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(mapping_frame, [pts], True, (0, 0, 255), 2)

                if len(second_player_points) == 4:
                    pts = np.array(second_player_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(mapping_frame, [pts], True, (255, 0, 0), 2)

                if len(net_points) == 4:
                    pts = np.array(net_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(mapping_frame, [pts], True, (0, 255, 0), 2)

                cv2.imshow('Map Court Boundaries', mapping_frame)

        # Set the callback function
        cv2.setMouseCallback('Map Court Boundaries', click_callback)

        # Display instructions
        instruction_frame = mapping_frame.copy()
        cv2.putText(instruction_frame, "Click to mark court boundaries:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "- First 4 clicks: First player area (red)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(instruction_frame, "- Next 4 clicks: Second player area (blue)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(instruction_frame, "- Last 4 clicks: Net area (green)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(instruction_frame, "Press 'S' to save or 'ESC' to cancel", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Map Court Boundaries', instruction_frame)

        # Wait for user to finish mapping or cancel
        while True:
            cv2.imshow('Map Court Boundaries', mapping_frame)
            key = cv2.waitKey(1) & 0xFF

            # Save and exit on 'S' key
            if key == ord('s') and len(first_player_points) == 4 and len(second_player_points) == 4 and len(
                    net_points) == 4:
                cv2.destroyWindow('Map Court Boundaries')

                # Save the points to config.ini
                self.save_court_points(first_player_points, second_player_points, net_points, points)
                return True

            # Cancel on ESC key
            elif key == 27:
                cv2.destroyWindow('Map Court Boundaries')
                return False

        cap.release()
        return False

    def save_court_points(self, first_player_points, second_player_points, net_points, crop_points):
        # Adjust for crop offset
        offset_y, offset_x = crop_points[0][1], crop_points[0][0]

        # Create config parser
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Update the stadium section
        if 'stadium' not in config:
            config['stadium'] = {}

        # Save first player points
        for i, point in enumerate(first_player_points, 1):
            x, y = point
            # Add crop offset to get original coordinates
            config['stadium'][f'first_player_point_{i}'] = f"{x + offset_x},{y + offset_y}"

        # Save second player points
        for i, point in enumerate(second_player_points, 1):
            x, y = point
            config['stadium'][f'second_player_point_{i}'] = f"{x + offset_x},{y + offset_y}"

        # Save net points
        for i, point in enumerate(net_points, 1):
            x, y = point
            config['stadium'][f'net_point_{i}'] = f"{x + offset_x},{y + offset_y}"

        # Write to config file
        with open('config.ini', 'w') as configfile:
            config.write(configfile)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
