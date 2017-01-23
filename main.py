import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

import calibrate
import undistort_transform
import thresholding


def calibration_parameters():
    if args.calibration_file == None:
        calibrate_dict = calibrate.calibrate()
        return calibrate_dict


def image_pipeline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    calibrate_dict = calibration_parameters()

    geometrical = undistort_transform.UndistortTransform(calibrate_dict, source_points, destination_points)
    gray = geometrical.undistort(gray)

    thresholder = thresholding.Thresholder(args.grad_mag_threshold, args.dir_threshold,
                                           args.color_threshold, args.sobel_kernel)

    mag_binary = thresholder.mag_thresh(gray)
    dir_binary = thresholder.dir_thresh(gray)

    mag_dir_binary = np.zeros_like(mag_binary)
    mag_dir_binary[(mag_binary == 1) | (dir_binary == 1)] = 1
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    if args.color_threshold_channel == 'H':
        channel = hsv_image[:,:,0]
    if args.color_threshold_channel == 'L':
        channel = hsv_image[:,:,1]
    if args.color_threshold_channel == 'S':
        channel = hsv_image[:,:,2]

    color_binary= thresholder.color_thresh(channel)

    combined_binary = np.zeros_like(color_binary)
    combined_binary[(mag_dir_binary == 1) | (color_binary == 1)] = 1

    combined_binary, Minv = geometrical.perspective_transform(combined_binary)
    return combined_binary


def video_pipeline():
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Finder")
    parser.add_argument('--calibration_file', type=str, default=None)
    parser.add_argument('--color_threshold', type=tuple, default=(170, 255))
    parser.add_argument('--grad_mag_threshold', type=tuple, default=(30, 100))
    parser.add_argument('--dir_threshold', type=tuple, default=(0.7, 1.3))
    parser.add_argument('--color_threshold_channel', type=str, default='S')
    parser.add_argument('--sobel_kernel', type=int, default=3)
    args = parser.parse_args()

    image = cv2.imread("./test_images/test1.jpg")
    imshape = image.shape
    source_points = np.float32([[0.47 * imshape[1], 0.60 * imshape[0]], [0.15 * imshape[1], imshape[0]],
                          [0.90 * imshape[1], imshape[0]], [0.52 * imshape[1], 0.60 * imshape[0]]])
    destination_points = np.float32([[100, 0], [100, 720], [920, 720], [920, 0]])
    processed = image_pipeline(image)

    plt.imshow(processed, cmap='gray')
    plt.show()