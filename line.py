import numpy as np
import cv2


class Line:
    def __init__(self):
        self.detected = False
        self.recent_x = []
        self.best_x = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.all_x = None
        self.all_y = None

class LineFinder:
    def __init__(self):
        return

    def histogram_window(self):
        return

    def measure_curvature(self):
        return