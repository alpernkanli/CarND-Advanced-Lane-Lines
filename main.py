import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

import calibrate
import undistort_transform
import thresholding
import line


def calibration_parameters():
    if args.calibration_file == None:
        calibrate_dict = calibrate.calibrate()
        return calibrate_dict


def image_pipeline(image, source_points, destination_points):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    calibrate_dict = calibration_parameters()

    geometrical = undistort_transform.UndistortTransform(calibrate_dict, source_points, destination_points)

    s = geometrical.undistort(hls_image[:, :, 2])
    r = geometrical.undistort(image[:, :, 2])

    thresholder = thresholding.Thresholder(args.grad_mag_threshold, args.dir_threshold,
                                           args.color_threshold)

    s_binary = thresholder.color_thresh(s, (100, 255))
    r_binary = thresholder.color_thresh(r, (100, 255))

    sr_combined = np.zeros_like(s_binary)
    sr_combined[(s_binary == 1) & (r_binary == 1)] = 1

    x_binary = thresholder.abs_sobel_thresh(r, 'x', 3, (30, 150))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(x_binary == 1) | (sr_combined == 1)] = 1
    mask = np.zeros_like(combined_binary)
    vertices = np.array([[(630, 420), (150, 720), (1180, 720), (750, 420)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (255))
    combined_binary = cv2.bitwise_and(combined_binary, mask)

    combined_binary, Minv = geometrical.perspective_transform(combined_binary)
    return combined_binary


def video_pipeline():
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Finder")
    parser.add_argument('--calibration_file', type=str, default=None)
    parser.add_argument('--color_threshold', type=tuple, default=(80, 255))
    parser.add_argument('--grad_mag_threshold', type=tuple, default=(120, 255))
    parser.add_argument('--dir_threshold', type=tuple, default=(0.6, 1.6))
    parser.add_argument('--color_threshold_channel', type=str, default='S')
    parser.add_argument('--sobel_kernel', type=int, default=3)
    args = parser.parse_args()

    image = cv2.imread("./test_images/test6.jpg")

    source_points = np.float32([[578, 460], [210, 720], [1128, 720], [710, 460]])
    destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    processed = image_pipeline(image, source_points, destination_points)

    plt.imshow(processed, cmap='gray')
    plt.show()
