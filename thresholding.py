import numpy as np
import cv2


class Thresholder:
    def __init__(self, grad_threshold, dir_threshold, sobel_kernel):
        self.grad_threshold = grad_threshold
        self.dir_threshold = dir_threshold

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        if orient == 'y':
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.abs(sobel)
        scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled)
        grad_binary[(scaled > thresh[0]) & (scaled < thresh[1])] = 1
        return grad_binary

    def mag_thresh(self, image, sobel_kernel=5):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        magnitude = np.sqrt(sobelx**2 +  sobely**2)
        scaled = np.uint8(255*magnitude/np.max(magnitude))
        mag_binary = np.zeros_like(scaled)
        mag_binary[(scaled > self.grad_threshold[0]) & (scaled < self.grad_threshold[1])] = 1
        return mag_binary

    def dir_thresh(self, image, sobel_kernel=5):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        absgraddir = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir > self.dir_threshold[0]) & (absgraddir < self.dir_threshold[1])] = 1
        return dir_binary

    def color_thresh(self, image, threshold):
        color_binary = np.zeros_like(image)
        color_binary[(image > threshold[0]) & (image < threshold[1])] = 1
        return color_binary

