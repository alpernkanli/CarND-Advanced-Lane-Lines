import numpy as np
import cv2
import pickle


class UndistortTransform:
    def __init__(self, calibrate_dict, source_points, destination_points):
        self.calibrate_dict = calibrate_dict
        self.source_points = source_points
        self.destination_points = destination_points

    def undistort(self, image):
        undist = cv2.undistort(image, self.calibrate_dict["mtx"],
                                self.calibrate_dict["dist"], None,
                                self.calibrate_dict["mtx"])
        return undist

    def perspective_transform(self, image):
        M = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        Minv = cv2.getPerspectiveTransform(self.destination_points, self.source_points)
        warped = cv2.warpPerspective(image, M, image.shape[0:2], flags=cv2.INTER_LINEAR)

        return warped, Minv