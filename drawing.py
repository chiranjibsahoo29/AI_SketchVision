import cv2
import numpy as np


class DrawingBoard:

    def __init__(self, width=640, height=480):
        self.canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        self.previous_point = None
        self.eraser_mode = False


        self.colors = [
            ("RED", (0, 0, 255)),
            ("GREEN", (0, 255, 0)),
            ("BLUE", (255, 0, 0)),
            ("YELLOW", (0, 255, 255)),
            ("PURPLE", (255, 0, 255)),
            ("WHITE", (255, 255, 255))
        ]

        self.current_color = (255, 255, 255)
        self.current_color_name = "WHITE"


        self.history = []
        self.max_history = 20


    def save_history(self):
        self.history.append(self.canvas.copy())

        if len(self.history) > self.max_history:
            self.history.pop(0)


    def draw(self, point):

        if point is None:
            self.previous_point = None
            return

        x, y = point


        if self.previous_point is None:
            self.save_history()
            self.previous_point = (x, y)
            return

        px, py = self.previous_point
        distance = np.hypot(x - px, y - py)


        MAX_STROKE_JUMP = 45

        if distance > MAX_STROKE_JUMP:
            self.previous_point = (x, y)
            return


        if distance < 2:
            return


        smooth_factor = 0.65
        smooth_x = int(px + (x - px) * smooth_factor)
        smooth_y = int(py + (y - py) * smooth_factor)

        if self.eraser_mode:
            cv2.line(
                self.canvas,
                (px, py),
                (smooth_x, smooth_y),
                (0, 0, 0),
                40,
                cv2.LINE_AA
            )
        else:
            cv2.line(
                self.canvas,
                (px, py),
                (smooth_x, smooth_y),
                self.current_color,
                5,
                cv2.LINE_AA
            )

        self.previous_point = (smooth_x, smooth_y)


    def undo(self):
        if self.history:
            self.canvas = self.history.pop()

        self.previous_point = None


    def clear(self):
        if np.any(self.canvas):
            self.save_history()

        self.canvas[:] = 0
        self.previous_point = None


    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        self.previous_point = None


    def set_color(self, name, color):
        self.current_color_name = name
        self.current_color = color
        self.eraser_mode = False
        self.previous_point = None
