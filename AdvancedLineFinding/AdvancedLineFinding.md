
# Advanced Lane Finding


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
%matplotlib inline
```

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I am heavily inspired by it. :)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the function calibrate():  
By defining chessboard corner counts in two dimensions, object points and image points (actually the corners from the cv2.findChessboardCorners()), and by using these parameters in cv2.calibrateCamera, the camera calibration parameters are extracted.

In the function undistort():  
Simply using cv2.undistort with the parameters which are found above, I undistort the images.


```python
def calibrate():
    chessboard_list = os.listdir("./camera_cal")
    nx = 9
    ny = 6

    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for chessboard in chessboard_list:

        img = cv2.imread("./camera_cal/" + chessboard)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    calibration_parameters = {}
    calibration_parameters["mtx"] = mtx
    calibration_parameters["dist"] = dist
    pickle.dump(calibration_parameters, open("calibration.p", "wb"))
    return mtx, dist
```


```python
mtx, dist = calibrate()
```


```python
def undistort(image, mtx, dist):
    undist = cv2.undistort(image, mtx,
                            dist, None,
                            mtx)
    return undist
```

And here is an example for the distortion corrected chessboard image:


```python
image = cv2.imread("./camera_cal/calibration3.jpg")
undistorted_image = undistort(image, mtx, dist)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title('Original Chessboard Image')
ax2.imshow(undistorted_image)
ax2.set_title('Undistorted Chessboard Image')
```




    <matplotlib.text.Text at 0x7f8836de92b0>




![png](output_10_1.png)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used the combination of sobel thresholding on x axis, s channel thresholding, and blue channel thresholding. Also magniture and direction thresholding is tried, but it made the algorithm slower without too much contribution.

The helper functions which are implemented are below:


```python
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 +  sobely**2)
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mag_binary

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def color_thresh(image, thresh=(0, 255)):
    color_binary = np.zeros_like(image)
    color_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return color_binary
```

And here is the example for the thresholding operation on single image pipeline:


```python
image = cv2.imread('./test_images/test2.jpg')
undist = undistort(image, mtx, dist)
undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
plt.imshow(undist)
```




    <matplotlib.image.AxesImage at 0x7f8836d07eb8>




![png](output_14_1.png)



```python
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

s = undistort(hls_image[:,:,2], mtx, dist)
b = undistort(image[:,:,0], mtx, dist)
gray = undistort(gray, mtx, dist)

s_binary = color_thresh(s, (160, 255))
b_binary = color_thresh(b, (200, 255))

color_combined = np.zeros_like(s_binary)

x_binary = abs_sobel_thresh(gray, 'x', 15, (70, 180))
color_combined[(s_binary == 1) | (x_binary == 1) | (b_binary == 1)] = 1

plt.imshow(color_combined, cmap='gray')
# plt.imshow(s_binary, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f88301ed668>




![png](output_15_1.png)



```python
hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
s = undistort(hls_image[:,:,2], mtx, dist)
r = undistort(image[:,:,0], mtx, dist)

s_binary = color_thresh(s, (190, 255))
r_binary = color_thresh(r, (20, 255))
    

x_binary = abs_sobel_thresh(r, 'x', 13, (20, 100))
    
    #combined_all = np.zeros_like(sr_combined)
combined_all = np.zeros_like(s_binary)
    
    #combined_all[(sr_combined == 1) | (x_binary == 1)] = 1
combined_all[(s_binary == 1) | (x_binary == 1)] = 1
    
vertices = np.array([[(570, 460), (200, 720), (1240, 720), (740, 460)]])
mask = np.zeros_like(combined_all)
cv2.fillPoly(mask, vertices, 255)
color_combined = cv2.bitwise_and(combined_all, mask)
plt.imshow(color_combined, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f88301bf588>




![png](output_16_1.png)



```python

```

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is implemented in the code block below, perpective_transform(). Inverse transform is stored for unwarping image afterwards


```python
def perspective_transform(image, source_points, destination_points):
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    Minv = cv2.getPerspectiveTransform(destination_points, source_points)
    warped = cv2.warpPerspective(image, M, (1280, 720), flags=cv2.INTER_LINEAR)

    return warped, Minv
```

The source points are chosen by inspection (by finding a trapezoid with side lines parallel to the straight lane lines roughly).

And here is an example for the perspective transform:


```python
source_points = np.float32([[579, 460], [210, 720], [1128, 720], [706, 460]])
destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

warped, Minv = perspective_transform(color_combined, source_points, destination_points)
plt.imshow(warped, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f883012a160>




![png](output_21_1.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Firstly, the approximate lane line positions (basis positions) are found by finding the histogram peaks:


```python
hist = np.sum(warped[warped.shape[0]//2:, :], axis=0)
plt.plot(hist)

left_peak = np.argmax(hist[:warped.shape[1]//2])
right_peak = np.argmax(hist[warped.shape[1]//2:]) + warped.shape[1]//2

print("Left peak is on: ", left_peak)
print("Right peak is on: ", right_peak)
```

    Left peak is on:  369
    Right peak is on:  959



![png](output_23_1.png)


After that, the polynomial points are defined with the sliding windows search which is performed on the image:


```python
out_img = np.dstack((warped, warped, warped))*255


nwindows = 9
window_height = warped.shape[0] // nwindows

nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = left_peak
rightx_current = right_peak

margin = 100
minpix = 50

left_lane_inds = []
right_lane_inds = []


# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped.shape[0] - (window+1)*window_height
    win_y_high = warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
```

By using np.polyfit function, polynomial fits are generated for the lines:


```python
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

To extrapolate the polynomials along the road, and generating the x and y values for plotting, the polynomial fit results are used:


```python

# Generate x and y values for plotting
fity = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(fit_leftx, fity, color='yellow')
plt.plot(fit_rightx, fity, color='yellow')

plt.xlim(0, 1280)
plt.ylim(720, 0)
```




    (720, 0)




![png](output_29_1.png)


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This time, the polynomial fit is performed with the real world values by approximating length by pixel values.

It is close to 1 km in this image, as expected.


```python
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700
y_eval = np.max(fity)*ym_per_pix
    
left_fit_real = np.polyfit(fity*ym_per_pix, fit_leftx*xm_per_pix, 2)
right_fit_real = np.polyfit(fity*ym_per_pix, fit_rightx*xm_per_pix, 2)

# Find the curve radii
left_roc = ((1 + (2*left_fit_real[0]*y_eval*ym_per_pix + left_fit_real[1])**2)**1.5) \
                             /np.absolute(2*left_fit_real[0])

right_roc = ((1 + (2*right_fit_real[0]*y_eval*ym_per_pix + right_fit_real[1])**2)**1.5) \
                             /np.absolute(2*right_fit_real[0])

radius_of_curvature = (left_roc + right_roc) / 2
print("The radius of curvature is: ", radius_of_curvature)
```

    The radius of curvature is:  933.09963193


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Below is the shape drawn by using cv2.fillPoly with the generated x and y values above.


```python
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)

color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

plt.imshow(color_warp)
```




    <matplotlib.image.AxesImage at 0x7f8826fc4550>




![png](output_33_1.png)


By using the inverse perspective transform matrix which is found earlier, the image found above is unwarped. And it is superposed on the original image (undistorted version of it).


```python
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (1280, 720)) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

# Add radius of curvature and lane position
left_line_base_pos = (1280 / 2 - left_peak) * 3.7 / 700
right_line_base_pos = (1280 / 2 - right_peak) * 3.7 / 700

lane_offset = (left_line_base_pos + right_line_base_pos) / 2
result = cv2.putText(result, "Curvature radius: " + str(radius_of_curvature), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
result = cv2.putText(result, "Lane position: " + str(lane_offset), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7f8826fa8d30>




![png](output_35_1.png)


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

All the code which is presented above is refactored and tidied up. Also, object oriented approach is used for convenience. For smoothing and keeping track of the values over the video, Line class is implemented.

Moreover, to increase performance in terms of speed, optimized search is implemented to use after lines are found.

Now, the code blocks will be explained.


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
%matplotlib inline
```


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def calibrate():
    chessboard_list = os.listdir("./camera_cal")
    nx = 9
    ny = 6

    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for chessboard in chessboard_list:

        img = cv2.imread("./camera_cal/" + chessboard)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    calibration_parameters = {}
    calibration_parameters["mtx"] = mtx
    calibration_parameters["dist"] = dist
    pickle.dump(calibration_parameters, open("calibration.p", "wb"))
    return mtx, dist
```


```python
def undistort(image, mtx, dist):
    undist = cv2.undistort(image, mtx,
                            dist, None,
                            mtx)
    return undist
```


```python
def perspective_transform(image, source_points, destination_points):
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    Minv = cv2.getPerspectiveTransform(destination_points, source_points)
    warped = cv2.warpPerspective(image, M, (1280, 720), flags=cv2.INTER_LINEAR)

    return warped, Minv
```


```python
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx**2 +  sobely**2)
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mag_binary

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def color_thresh(image, thresh=(0, 255)):
    color_binary = np.zeros_like(image)
    color_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return color_binary
```

The container class "Parameters" for convenience:


```python
# Container class for geometrical parameters
class Parameters:
    def __init__(self):
        # Calibration parameters
        self.mtx = None
        self.dst = None
        # Perspective transform parameters
        self.M = None
        self.Minv = None
        self.warped_shape = None
        # Masking parameter
        self.vertices = None
        self.mask = None
```

The "Line" class has the parameters for buffering and smoothing operations.

Also, it has three functions "update_fit", "update_peak", "update_metrics" to update and store there values in lists.


```python
class Line():
    def __init__(self):
        # Maximum of the buffer
        self.limit = 30
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None 
        #polynomial coefficients of the last n fits of the line
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])] 
        #polynomial coefficients for the most recent metric fit
        self.current_metric_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = []
        self.best_roc = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Current peak of histogram
        self.peak = None
        # Recent peaks of histogram
        self.recent_peaks = []
        # Average of the peaks
        self.best_peak = None
        # Lane indices
        self.line_inds = []
        # Current lane indices
        self.current_line_inds = None
        
    def update_fit(self, values, metric_values, xfit):
        self.recent_fits.append(values)
        self.current_fit = values
        self.recent_xfitted.append(xfit)
        if len(self.recent_fits) > self.limit:
            self.recent_fits = self.recent_fits[1:]
            self.recent_xfitted = self.recent_xfitted[1:]
        self.best_fit = np.mean(self.recent_fits, axis=0)
        self.current_metric_fit = metric_values
        self.bestx = np.mean(self.recent_xfitted, axis=0)
    
    def update_peak(self, value):
        self.peak = value
        self.recent_peaks.append(value)
        if len(self.recent_peaks) > self.limit:
            self.recent_peaks = self.recent_peaks[1:]
        self.best_peak = np.mean(self.recent_peaks, axis=0)
        
    def update_metrics(self, curverad):
        self.radius_of_curvature.append(curverad)
        if len(self.radius_of_curvature) > self.limit:
            self.radius_of_curvature = self.radius_of_curvature[1:]
            
        self.best_roc = np.mean(self.radius_of_curvature, axis=0)
```


```python
left_line = Line()
right_line = Line()
```


```python
parameters = Parameters()
parameters.mtx, parameters.dst = calibrate()
parameters.image_size = (1280, 720)
```

The image procesing (undistorting, filtering, masking and warping) operations which are mentioned in the single image pipeline, are implemented here for the video:


```python
def process_image2(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = undistort(hls_image[:,:,2], parameters.mtx, parameters.dst)
    b = undistort(image[:,:,2], parameters.mtx, parameters.dst)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = undistort(gray, parameters.mtx, parameters.dst)
    s_binary = color_thresh(s, (160, 255))
    b_binary = color_thresh(b, (200, 255))
    color_combined = np.zeros_like(s_binary)
    x_binary = abs_sobel_thresh(gray, 'x', 15, (70, 180)) 
    combined_all = np.zeros_like(s_binary)
    combined_all[(s_binary == 1) | (x_binary == 1) | (b_binary == 1)] = 1
    
    vertices = np.array([[(570, 460), (200, 720), (1240, 720), (740, 460)]])
    mask = np.zeros_like(combined_all)
    cv2.fillPoly(mask, vertices, 255)
    combined_all = cv2.bitwise_and(combined_all, mask)


    source_points = np.float32([[582, 460], [210, 720], [1128, 720], [704, 460]])
    destination_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    warped, parameters.Minv = perspective_transform(combined_all, source_points, destination_points)
    
    return warped
```

This function finds the basis peak values and updates the peak list of the line objects:


```python
def set_base_peak(warped):
    hist = np.sum(warped[720//2:, :], axis=0)
    
    left_line.update_peak(np.argmax(hist[:1280//2]))
    right_line.update_peak(np.argmax(hist[1280//2:]) + 1280//2)
```


```python
def search_in_frame(warped, nonzerox, nonzeroy):
    nwindows = 9
    window_height = warped.shape[0] // nwindows

    leftx_current = left_line.peak
    rightx_current = right_line.peak

    margin = 70
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []


    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    left_line_inds = np.concatenate(left_lane_inds)
    right_line_inds = np.concatenate(right_lane_inds)
    return left_line_inds, right_line_inds
```

This function performs an optimized search once the lines are found:


```python
def search_in_margin(image, nonzerox, nonzeroy):
    margin = 70
    left_line_inds = ((nonzerox > (left_line.best_fit[0]*(nonzeroy**2) 
                                   + left_line.best_fit[1]*nonzeroy 
                                   + left_line.best_fit[2] - margin)) 
                      & (nonzerox < (left_line.best_fit[0]*(nonzeroy**2) 
                                     + left_line.best_fit[1]*nonzeroy 
                                     + left_line.best_fit[2] + margin))).nonzero()[0]
    right_line_inds = ((nonzerox > (right_line.best_fit[0]*(nonzeroy**2) 
                                    + right_line.best_fit[1]*nonzeroy 
                                    + right_line.best_fit[2] - margin)) 
                       & (nonzerox < (right_line.best_fit[0]*(nonzeroy**2) 
                                      + right_line.best_fit[1]*nonzeroy 
                                      + right_line.best_fit[2] + margin))).nonzero()[0]

    return left_line_inds, right_line_inds
```

This function calculates polynomial fits and the radius of curvature for each line.


```python
def calculate_metrics(line):
    ploty = np.linspace(0, 719, 720)
    fit = np.polyfit(line.ally, line.allx, 2)
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    # Fit second order polynomial with real world values
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    y_eval = np.max(line.ally)*ym_per_pix
    
    fit_real = np.polyfit(line.ally*ym_per_pix, line.allx*xm_per_pix, 2)
    
    
    # Update fits of the lines
    line.update_fit(fit, fit_real, fitx)
    fit = line.best_fit
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    # Find the curve radii
    radius_of_curvature = ((1 + (2*fit_real[0]*y_eval*ym_per_pix + fit_real[1])**2)**1.5) \
                             /np.absolute(2*fit_real[0])
    line.update_metrics(radius_of_curvature)
```

And this function visualizes the unwarped version of the image with the radius of curvature and lane position values:


```python
def visualize(image, warped):
    ploty = np.linspace(0, 719, 720)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    undist = undistort(image, parameters.mtx, parameters.dst)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, parameters.Minv, (1280, 720)) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # Display values
    #curverad = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
    curverad = (left_line.best_roc + right_line.best_roc) / 2
    lane_offset = (left_line.line_base_pos + right_line.line_base_pos) / 2
    result = cv2.putText(result, "Curvature radius: " + str(curverad), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    result = cv2.putText(result, "Lane position: " + str(lane_offset), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return result
```

And this is the implementation for the video frame processing:


```python
def pipeline(image):
    check = False
    # Undistorting, filtering and perspective transform
    warped = process_image2(image)

    # For visualization
    out_img = np.dstack((warped, warped, warped))*255
    
    # Identifying nonzero x and y pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Finding basis histogram peaks
    set_base_peak(warped)
    
    left_line_inds = []
    right_line_inds = []
    
    # Sliding windows search, in frame or in small margin
    if (left_line.detected and right_line.detected):
        left_line_inds, right_line_inds = search_in_margin(image, nonzerox, nonzeroy)
    else:
        left_line_inds, right_line_inds = search_in_frame(image, nonzerox, nonzeroy)
        
    left_line.detected = False
    right_line.detected = False
    
    if len(left_line_inds) > 0 and len(right_line_inds) > 0:
        
        left_line.allx = nonzerox[left_line_inds]
        right_line.allx = nonzerox[right_line_inds]
        left_line.ally = nonzeroy[left_line_inds]
        right_line.ally = nonzeroy[right_line_inds]
        
        if len(left_line.ally) > 0  and len(left_line.allx) > 0:
            left_line.detected = True
            calculate_metrics(left_line)
        if len(right_line.ally) > 0  and len(right_line.allx) > 0:
            right_line.detected = True
            calculate_metrics(right_line)
        check = True
    left_line.line_base_pos = (1280 / 2 - left_line.best_peak) * 3.7 / 700
    right_line.line_base_pos = (1280 / 2 - right_line.best_peak) * 3.7 / 700
    
    result = visualize(image, warped)
    
    return result
```


```python
video_output = "output1.mp4"
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(pipeline)
%time clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video output1.mp4
    [MoviePy] Writing video output1.mp4


    
      0%|          | 0/1261 [00:00<?, ?it/s][A
      0%|          | 1/1261 [00:00<09:40,  2.17it/s][A
      0%|          | 2/1261 [00:00<08:50,  2.37it/s][A
      0%|          | 3/1261 [00:01<08:38,  2.42it/s][A
      0%|          | 4/1261 [00:01<09:11,  2.28it/s][A
      0%|          | 5/1261 [00:02<08:36,  2.43it/s][A
      0%|          | 6/1261 [00:02<08:12,  2.55it/s][A
      1%|          | 7/1261 [00:02<08:37,  2.42it/s][A
      1%|          | 8/1261 [00:03<08:43,  2.40it/s][A
      1%|          | 9/1261 [00:03<08:26,  2.47it/s][A
      1%|          | 10/1261 [00:04<08:14,  2.53it/s][A
      1%|          | 11/1261 [00:04<08:09,  2.55it/s][A
      1%|          | 12/1261 [00:04<08:13,  2.53it/s][A
      1%|          | 13/1261 [00:05<07:54,  2.63it/s][A
      1%|          | 14/1261 [00:05<08:56,  2.33it/s][A
      1%|          | 15/1261 [00:06<08:46,  2.37it/s][A
      1%|▏         | 16/1261 [00:06<08:59,  2.31it/s][A
      1%|▏         | 17/1261 [00:06<08:57,  2.31it/s][A
      1%|▏         | 18/1261 [00:07<08:57,  2.31it/s][A
      2%|▏         | 19/1261 [00:07<09:08,  2.27it/s][A
      2%|▏         | 20/1261 [00:08<08:57,  2.31it/s][A
      2%|▏         | 21/1261 [00:08<09:08,  2.26it/s][A
      2%|▏         | 22/1261 [00:09<08:55,  2.31it/s][A
      2%|▏         | 23/1261 [00:09<08:27,  2.44it/s][A
      2%|▏         | 24/1261 [00:09<08:01,  2.57it/s][A
      2%|▏         | 25/1261 [00:10<08:17,  2.48it/s][A
      2%|▏         | 26/1261 [00:10<08:27,  2.43it/s][A
      2%|▏         | 27/1261 [00:11<08:25,  2.44it/s][A
      2%|▏         | 28/1261 [00:11<07:55,  2.60it/s][A
      2%|▏         | 29/1261 [00:11<07:27,  2.75it/s][A
      2%|▏         | 30/1261 [00:12<07:31,  2.73it/s][A
      2%|▏         | 31/1261 [00:12<07:17,  2.81it/s][A
      3%|▎         | 32/1261 [00:12<07:10,  2.86it/s][A
      3%|▎         | 33/1261 [00:13<07:43,  2.65it/s][A
      3%|▎         | 34/1261 [00:13<07:38,  2.67it/s][A
      3%|▎         | 35/1261 [00:13<07:27,  2.74it/s][A
      3%|▎         | 36/1261 [00:14<07:38,  2.67it/s][A
      3%|▎         | 37/1261 [00:14<07:05,  2.88it/s][A
      3%|▎         | 38/1261 [00:14<06:53,  2.95it/s][A
      3%|▎         | 39/1261 [00:15<07:02,  2.89it/s][A
      3%|▎         | 40/1261 [00:15<06:46,  3.01it/s][A
      3%|▎         | 41/1261 [00:15<06:48,  2.98it/s][A
      3%|▎         | 42/1261 [00:16<06:49,  2.98it/s][A
      3%|▎         | 43/1261 [00:16<07:59,  2.54it/s][A
      3%|▎         | 44/1261 [00:17<07:34,  2.68it/s][A
      4%|▎         | 45/1261 [00:17<07:24,  2.74it/s][A
      4%|▎         | 46/1261 [00:17<07:05,  2.86it/s][A
      4%|▎         | 47/1261 [00:18<07:20,  2.76it/s][A
      4%|▍         | 48/1261 [00:18<07:16,  2.78it/s][A
      4%|▍         | 49/1261 [00:18<06:48,  2.97it/s][A
      4%|▍         | 50/1261 [00:19<06:30,  3.10it/s][A
      4%|▍         | 51/1261 [00:19<06:50,  2.94it/s][A
      4%|▍         | 52/1261 [00:19<06:49,  2.95it/s][A
      4%|▍         | 53/1261 [00:20<07:00,  2.87it/s][A
      4%|▍         | 54/1261 [00:20<07:00,  2.87it/s][A
      4%|▍         | 55/1261 [00:20<07:02,  2.85it/s][A
      4%|▍         | 56/1261 [00:21<07:08,  2.81it/s][A
      5%|▍         | 57/1261 [00:21<06:59,  2.87it/s][A
      5%|▍         | 58/1261 [00:21<07:00,  2.86it/s][A
      5%|▍         | 59/1261 [00:22<07:15,  2.76it/s][A
      5%|▍         | 60/1261 [00:22<06:58,  2.87it/s][A
      5%|▍         | 61/1261 [00:23<07:06,  2.81it/s][A
      5%|▍         | 62/1261 [00:23<07:04,  2.82it/s][A
      5%|▍         | 63/1261 [00:23<06:54,  2.89it/s][A
      5%|▌         | 64/1261 [00:24<07:08,  2.79it/s][A
      5%|▌         | 65/1261 [00:24<07:36,  2.62it/s][A
      5%|▌         | 66/1261 [00:25<08:22,  2.38it/s][A
      5%|▌         | 67/1261 [00:25<08:40,  2.30it/s][A
      5%|▌         | 68/1261 [00:25<08:15,  2.41it/s][A
      5%|▌         | 69/1261 [00:26<08:24,  2.36it/s][A
      6%|▌         | 70/1261 [00:26<08:47,  2.26it/s][A
      6%|▌         | 71/1261 [00:27<08:31,  2.33it/s][A
      6%|▌         | 72/1261 [00:27<08:06,  2.45it/s][A
      6%|▌         | 73/1261 [00:28<08:06,  2.44it/s][A
      6%|▌         | 74/1261 [00:28<07:54,  2.50it/s][A
      6%|▌         | 75/1261 [00:28<07:43,  2.56it/s][A
      6%|▌         | 76/1261 [00:29<07:36,  2.59it/s][A
      6%|▌         | 77/1261 [00:29<07:24,  2.67it/s][A
      6%|▌         | 78/1261 [00:29<07:20,  2.69it/s][A
      6%|▋         | 79/1261 [00:30<07:30,  2.63it/s][A
      6%|▋         | 80/1261 [00:30<07:18,  2.70it/s][A
      6%|▋         | 81/1261 [00:30<07:09,  2.75it/s][A
      7%|▋         | 82/1261 [00:31<06:55,  2.84it/s][A
      7%|▋         | 83/1261 [00:31<06:57,  2.82it/s][A
      7%|▋         | 84/1261 [00:32<07:06,  2.76it/s][A
      7%|▋         | 85/1261 [00:32<07:29,  2.62it/s][A
      7%|▋         | 86/1261 [00:32<07:44,  2.53it/s][A
      7%|▋         | 87/1261 [00:33<07:15,  2.70it/s][A
      7%|▋         | 88/1261 [00:33<07:23,  2.64it/s][A
      7%|▋         | 89/1261 [00:33<07:26,  2.63it/s][A
      7%|▋         | 90/1261 [00:34<07:30,  2.60it/s][A
      7%|▋         | 91/1261 [00:34<06:58,  2.80it/s][A
      7%|▋         | 92/1261 [00:35<07:10,  2.72it/s][A
      7%|▋         | 93/1261 [00:35<07:11,  2.71it/s][A
      7%|▋         | 94/1261 [00:35<06:51,  2.84it/s][A
      8%|▊         | 95/1261 [00:36<06:33,  2.96it/s][A
      8%|▊         | 96/1261 [00:36<06:53,  2.82it/s][A
      8%|▊         | 97/1261 [00:36<06:40,  2.90it/s][A
      8%|▊         | 98/1261 [00:37<06:58,  2.78it/s][A
      8%|▊         | 99/1261 [00:37<07:30,  2.58it/s][A
      8%|▊         | 100/1261 [00:37<07:27,  2.59it/s][A
      8%|▊         | 101/1261 [00:38<07:22,  2.62it/s][A
      8%|▊         | 102/1261 [00:38<07:25,  2.60it/s][A
      8%|▊         | 103/1261 [00:39<07:17,  2.64it/s][A
      8%|▊         | 104/1261 [00:39<07:19,  2.63it/s][A
      8%|▊         | 105/1261 [00:39<07:29,  2.57it/s][A
      8%|▊         | 106/1261 [00:40<07:50,  2.46it/s][A
      8%|▊         | 107/1261 [00:40<08:14,  2.34it/s][A
      9%|▊         | 108/1261 [00:41<07:55,  2.43it/s][A
      9%|▊         | 109/1261 [00:41<07:35,  2.53it/s][A
      9%|▊         | 110/1261 [00:41<07:38,  2.51it/s][A
      9%|▉         | 111/1261 [00:42<07:08,  2.69it/s][A
      9%|▉         | 112/1261 [00:42<07:02,  2.72it/s][A
      9%|▉         | 113/1261 [00:43<07:38,  2.51it/s][A
      9%|▉         | 114/1261 [00:43<07:32,  2.53it/s][A
      9%|▉         | 115/1261 [00:43<07:26,  2.57it/s][A
      9%|▉         | 116/1261 [00:44<07:17,  2.62it/s][A
      9%|▉         | 117/1261 [00:44<07:32,  2.53it/s][A
      9%|▉         | 118/1261 [00:45<07:33,  2.52it/s][A
      9%|▉         | 119/1261 [00:45<07:29,  2.54it/s][A
     10%|▉         | 120/1261 [00:45<07:31,  2.53it/s][A
     10%|▉         | 121/1261 [00:46<07:40,  2.48it/s][A
     10%|▉         | 122/1261 [00:46<07:20,  2.58it/s][A
     10%|▉         | 123/1261 [00:46<07:11,  2.64it/s][A
     10%|▉         | 124/1261 [00:47<07:16,  2.60it/s][A
     10%|▉         | 125/1261 [00:47<06:42,  2.82it/s][A
     10%|▉         | 126/1261 [00:48<07:00,  2.70it/s][A
     10%|█         | 127/1261 [00:48<06:42,  2.82it/s][A
     10%|█         | 128/1261 [00:48<06:58,  2.71it/s][A
     10%|█         | 129/1261 [00:49<07:06,  2.65it/s][A
     10%|█         | 130/1261 [00:49<07:13,  2.61it/s][A
     10%|█         | 131/1261 [00:50<07:25,  2.54it/s][A
     10%|█         | 132/1261 [00:50<07:27,  2.52it/s][A
     11%|█         | 133/1261 [00:50<07:25,  2.53it/s][A
     11%|█         | 134/1261 [00:51<08:04,  2.33it/s][A
     11%|█         | 135/1261 [00:51<07:55,  2.37it/s][A
     11%|█         | 136/1261 [00:52<07:57,  2.36it/s][A
     11%|█         | 137/1261 [00:52<07:56,  2.36it/s][A
     11%|█         | 138/1261 [00:53<08:40,  2.16it/s][A
     11%|█         | 139/1261 [00:53<08:23,  2.23it/s][A
     11%|█         | 140/1261 [00:53<08:07,  2.30it/s][A
     11%|█         | 141/1261 [00:54<08:00,  2.33it/s][A
     11%|█▏        | 142/1261 [00:54<07:52,  2.37it/s][A
     11%|█▏        | 143/1261 [00:55<07:48,  2.39it/s][A
     11%|█▏        | 144/1261 [00:55<08:00,  2.33it/s][A
     11%|█▏        | 145/1261 [00:56<07:48,  2.38it/s][A
     12%|█▏        | 146/1261 [00:56<07:51,  2.36it/s][A
     12%|█▏        | 147/1261 [00:56<07:23,  2.51it/s][A
     12%|█▏        | 148/1261 [00:57<06:53,  2.69it/s][A
     12%|█▏        | 149/1261 [00:57<07:02,  2.63it/s][A
     12%|█▏        | 150/1261 [00:57<06:51,  2.70it/s][A
     12%|█▏        | 151/1261 [00:58<06:34,  2.82it/s][A
     12%|█▏        | 152/1261 [00:58<06:42,  2.76it/s][A
     12%|█▏        | 153/1261 [00:58<06:38,  2.78it/s][A
     12%|█▏        | 154/1261 [00:59<06:50,  2.70it/s][A
     12%|█▏        | 155/1261 [00:59<06:49,  2.70it/s][A
     12%|█▏        | 156/1261 [01:00<06:42,  2.74it/s][A
     12%|█▏        | 157/1261 [01:00<06:51,  2.68it/s][A
     13%|█▎        | 158/1261 [01:00<07:01,  2.62it/s][A
     13%|█▎        | 159/1261 [01:01<07:05,  2.59it/s][A
     13%|█▎        | 160/1261 [01:01<07:09,  2.56it/s][A
     13%|█▎        | 161/1261 [01:01<06:51,  2.67it/s][A
     13%|█▎        | 162/1261 [01:02<06:34,  2.79it/s][A
     13%|█▎        | 163/1261 [01:02<06:45,  2.71it/s][A
     13%|█▎        | 164/1261 [01:03<06:41,  2.73it/s][A
     13%|█▎        | 165/1261 [01:03<06:27,  2.82it/s][A
     13%|█▎        | 166/1261 [01:03<06:25,  2.84it/s][A
     13%|█▎        | 167/1261 [01:04<06:31,  2.79it/s][A
     13%|█▎        | 168/1261 [01:04<06:08,  2.96it/s][A
     13%|█▎        | 169/1261 [01:04<06:11,  2.94it/s][A
     13%|█▎        | 170/1261 [01:05<06:34,  2.77it/s][A
     14%|█▎        | 171/1261 [01:05<06:34,  2.76it/s][A
     14%|█▎        | 172/1261 [01:05<06:25,  2.82it/s][A
     14%|█▎        | 173/1261 [01:06<06:32,  2.77it/s][A
     14%|█▍        | 174/1261 [01:06<06:17,  2.88it/s][A
     14%|█▍        | 175/1261 [01:06<06:29,  2.79it/s][A
     14%|█▍        | 176/1261 [01:07<06:17,  2.88it/s][A
     14%|█▍        | 177/1261 [01:07<06:24,  2.82it/s][A
     14%|█▍        | 178/1261 [01:08<06:52,  2.63it/s][A
     14%|█▍        | 179/1261 [01:08<06:43,  2.68it/s][A
     14%|█▍        | 180/1261 [01:08<06:24,  2.81it/s][A
     14%|█▍        | 181/1261 [01:09<06:20,  2.84it/s][A
     14%|█▍        | 182/1261 [01:09<06:24,  2.80it/s][A
     15%|█▍        | 183/1261 [01:09<06:03,  2.97it/s][A
     15%|█▍        | 184/1261 [01:10<06:24,  2.80it/s][A
     15%|█▍        | 185/1261 [01:10<06:08,  2.92it/s][A
     15%|█▍        | 186/1261 [01:10<06:14,  2.87it/s][A
     15%|█▍        | 187/1261 [01:11<06:03,  2.96it/s][A
     15%|█▍        | 188/1261 [01:11<06:09,  2.90it/s][A
     15%|█▍        | 189/1261 [01:11<06:11,  2.89it/s][A
     15%|█▌        | 190/1261 [01:12<05:59,  2.98it/s][A
     15%|█▌        | 191/1261 [01:12<05:59,  2.98it/s][A
     15%|█▌        | 192/1261 [01:12<05:45,  3.09it/s][A
     15%|█▌        | 193/1261 [01:13<05:49,  3.06it/s][A
     15%|█▌        | 194/1261 [01:13<05:58,  2.98it/s][A
     15%|█▌        | 195/1261 [01:13<05:51,  3.03it/s][A
     16%|█▌        | 196/1261 [01:14<06:06,  2.91it/s][A
     16%|█▌        | 197/1261 [01:14<05:59,  2.96it/s][A
     16%|█▌        | 198/1261 [01:14<05:47,  3.06it/s][A
     16%|█▌        | 199/1261 [01:15<06:09,  2.88it/s][A
     16%|█▌        | 200/1261 [01:15<06:17,  2.81it/s][A
     16%|█▌        | 201/1261 [01:15<06:15,  2.82it/s][A
     16%|█▌        | 202/1261 [01:16<06:20,  2.78it/s][A
     16%|█▌        | 203/1261 [01:16<05:57,  2.96it/s][A
     16%|█▌        | 204/1261 [01:16<05:45,  3.06it/s][A
     16%|█▋        | 205/1261 [01:17<05:48,  3.03it/s][A
     16%|█▋        | 206/1261 [01:17<05:47,  3.04it/s][A
     16%|█▋        | 207/1261 [01:17<05:39,  3.11it/s][A
     16%|█▋        | 208/1261 [01:18<05:34,  3.15it/s][A
     17%|█▋        | 209/1261 [01:18<05:20,  3.28it/s][A
     17%|█▋        | 210/1261 [01:18<05:28,  3.20it/s][A
     17%|█▋        | 211/1261 [01:18<05:20,  3.27it/s][A
     17%|█▋        | 212/1261 [01:19<05:12,  3.35it/s][A
     17%|█▋        | 213/1261 [01:19<05:00,  3.49it/s][A
     17%|█▋        | 214/1261 [01:19<04:46,  3.65it/s][A
     17%|█▋        | 215/1261 [01:20<05:28,  3.18it/s][A
     17%|█▋        | 216/1261 [01:20<05:40,  3.07it/s][A
     17%|█▋        | 217/1261 [01:20<05:28,  3.17it/s][A
     17%|█▋        | 218/1261 [01:21<05:10,  3.36it/s][A
     17%|█▋        | 219/1261 [01:21<05:06,  3.40it/s][A
     17%|█▋        | 220/1261 [01:21<04:51,  3.57it/s][A
     18%|█▊        | 221/1261 [01:21<04:50,  3.58it/s][A
     18%|█▊        | 222/1261 [01:22<04:55,  3.51it/s][A
     18%|█▊        | 223/1261 [01:22<04:57,  3.49it/s][A
     18%|█▊        | 224/1261 [01:22<04:42,  3.67it/s][A
     18%|█▊        | 225/1261 [01:23<04:45,  3.62it/s][A
     18%|█▊        | 226/1261 [01:23<04:42,  3.66it/s][A
     18%|█▊        | 227/1261 [01:23<04:47,  3.60it/s][A
     18%|█▊        | 228/1261 [01:23<04:47,  3.60it/s][A
     18%|█▊        | 229/1261 [01:24<04:48,  3.57it/s][A
     18%|█▊        | 230/1261 [01:24<04:48,  3.58it/s][A
     18%|█▊        | 231/1261 [01:24<04:47,  3.59it/s][A
     18%|█▊        | 232/1261 [01:24<04:40,  3.67it/s][A
     18%|█▊        | 233/1261 [01:25<04:33,  3.75it/s][A
     19%|█▊        | 234/1261 [01:25<04:23,  3.89it/s][A
     19%|█▊        | 235/1261 [01:25<04:24,  3.88it/s][A
     19%|█▊        | 236/1261 [01:25<04:22,  3.90it/s][A
     19%|█▉        | 237/1261 [01:26<04:21,  3.91it/s][A
     19%|█▉        | 238/1261 [01:26<04:23,  3.88it/s][A
     19%|█▉        | 239/1261 [01:26<04:10,  4.08it/s][A
     19%|█▉        | 240/1261 [01:26<04:18,  3.95it/s][A
     19%|█▉        | 241/1261 [01:27<04:10,  4.07it/s][A
     19%|█▉        | 242/1261 [01:27<04:11,  4.05it/s][A
     19%|█▉        | 243/1261 [01:27<04:08,  4.10it/s][A
     19%|█▉        | 244/1261 [01:27<04:05,  4.14it/s][A
     19%|█▉        | 245/1261 [01:28<04:55,  3.44it/s][A
     20%|█▉        | 246/1261 [01:28<05:03,  3.34it/s][A
     20%|█▉        | 247/1261 [01:28<05:00,  3.37it/s][A
     20%|█▉        | 248/1261 [01:29<04:51,  3.47it/s][A
     20%|█▉        | 249/1261 [01:29<04:33,  3.70it/s][A
     20%|█▉        | 250/1261 [01:29<04:28,  3.77it/s][A
     20%|█▉        | 251/1261 [01:29<04:14,  3.96it/s][A
     20%|█▉        | 252/1261 [01:30<04:18,  3.90it/s][A
     20%|██        | 253/1261 [01:30<04:21,  3.85it/s][A
     20%|██        | 254/1261 [01:30<04:19,  3.87it/s][A
     20%|██        | 255/1261 [01:30<04:23,  3.81it/s][A
     20%|██        | 256/1261 [01:31<04:13,  3.96it/s][A
     20%|██        | 257/1261 [01:31<04:10,  4.01it/s][A
     20%|██        | 258/1261 [01:31<04:03,  4.13it/s][A
     21%|██        | 259/1261 [01:31<04:18,  3.88it/s][A
     21%|██        | 260/1261 [01:32<04:09,  4.02it/s][A
     21%|██        | 261/1261 [01:32<04:12,  3.97it/s][A
     21%|██        | 262/1261 [01:32<04:09,  4.01it/s][A
     21%|██        | 263/1261 [01:32<04:23,  3.79it/s][A
     21%|██        | 264/1261 [01:33<04:17,  3.87it/s][A
     21%|██        | 265/1261 [01:33<04:16,  3.88it/s][A
     21%|██        | 266/1261 [01:33<04:05,  4.06it/s][A
     21%|██        | 267/1261 [01:33<04:05,  4.05it/s][A
     21%|██▏       | 268/1261 [01:34<03:55,  4.21it/s][A
     21%|██▏       | 269/1261 [01:34<04:08,  3.98it/s][A
     21%|██▏       | 270/1261 [01:34<04:07,  4.00it/s][A
     21%|██▏       | 271/1261 [01:34<04:15,  3.87it/s][A
     22%|██▏       | 272/1261 [01:35<04:06,  4.01it/s][A
     22%|██▏       | 273/1261 [01:35<04:10,  3.95it/s][A
     22%|██▏       | 274/1261 [01:35<04:06,  4.00it/s][A
     22%|██▏       | 275/1261 [01:35<04:20,  3.79it/s][A
     22%|██▏       | 276/1261 [01:36<04:10,  3.94it/s][A
     22%|██▏       | 277/1261 [01:36<04:12,  3.89it/s][A
     22%|██▏       | 278/1261 [01:36<04:14,  3.86it/s][A
     22%|██▏       | 279/1261 [01:37<04:18,  3.80it/s][A
     22%|██▏       | 280/1261 [01:37<04:15,  3.84it/s][A
     22%|██▏       | 281/1261 [01:37<04:20,  3.77it/s][A
     22%|██▏       | 282/1261 [01:37<04:18,  3.79it/s][A
     22%|██▏       | 283/1261 [01:38<04:18,  3.79it/s][A
     23%|██▎       | 284/1261 [01:38<04:13,  3.86it/s][A
     23%|██▎       | 285/1261 [01:38<04:04,  3.99it/s][A
     23%|██▎       | 286/1261 [01:38<04:11,  3.88it/s][A
     23%|██▎       | 287/1261 [01:39<04:06,  3.95it/s][A
     23%|██▎       | 288/1261 [01:39<04:00,  4.05it/s][A
     23%|██▎       | 289/1261 [01:39<04:10,  3.88it/s][A
     23%|██▎       | 290/1261 [01:39<04:09,  3.88it/s][A
     23%|██▎       | 291/1261 [01:40<04:24,  3.67it/s][A
     23%|██▎       | 292/1261 [01:40<04:17,  3.76it/s][A
     23%|██▎       | 293/1261 [01:40<04:24,  3.66it/s][A
     23%|██▎       | 294/1261 [01:40<04:28,  3.60it/s][A
     23%|██▎       | 295/1261 [01:41<04:32,  3.54it/s][A
     23%|██▎       | 296/1261 [01:41<04:21,  3.68it/s][A
     24%|██▎       | 297/1261 [01:41<04:19,  3.72it/s][A
     24%|██▎       | 298/1261 [01:42<04:14,  3.79it/s][A
     24%|██▎       | 299/1261 [01:42<04:22,  3.67it/s][A
     24%|██▍       | 300/1261 [01:42<04:15,  3.77it/s][A
     24%|██▍       | 301/1261 [01:42<04:15,  3.76it/s][A
     24%|██▍       | 302/1261 [01:43<04:15,  3.75it/s][A
     24%|██▍       | 303/1261 [01:43<04:28,  3.56it/s][A
     24%|██▍       | 304/1261 [01:43<04:14,  3.76it/s][A
     24%|██▍       | 305/1261 [01:43<04:20,  3.68it/s][A
     24%|██▍       | 306/1261 [01:44<04:29,  3.54it/s][A
     24%|██▍       | 307/1261 [01:44<04:16,  3.71it/s][A
     24%|██▍       | 308/1261 [01:44<04:39,  3.41it/s][A
     25%|██▍       | 309/1261 [01:45<04:27,  3.56it/s][A
     25%|██▍       | 310/1261 [01:45<04:15,  3.72it/s][A
     25%|██▍       | 311/1261 [01:45<04:16,  3.70it/s][A
     25%|██▍       | 312/1261 [01:45<04:13,  3.75it/s][A
     25%|██▍       | 313/1261 [01:46<04:10,  3.79it/s][A
     25%|██▍       | 314/1261 [01:46<04:20,  3.63it/s][A
     25%|██▍       | 315/1261 [01:46<04:23,  3.58it/s][A
     25%|██▌       | 316/1261 [01:47<04:33,  3.46it/s][A
     25%|██▌       | 317/1261 [01:47<04:44,  3.32it/s][A
     25%|██▌       | 318/1261 [01:47<04:46,  3.29it/s][A
     25%|██▌       | 319/1261 [01:47<04:36,  3.41it/s][A
     25%|██▌       | 320/1261 [01:48<04:30,  3.48it/s][A
     25%|██▌       | 321/1261 [01:48<04:37,  3.39it/s][A
     26%|██▌       | 322/1261 [01:48<04:20,  3.61it/s][A
     26%|██▌       | 323/1261 [01:49<04:22,  3.57it/s][A
     26%|██▌       | 324/1261 [01:49<04:19,  3.62it/s][A
     26%|██▌       | 325/1261 [01:49<04:17,  3.63it/s][A
     26%|██▌       | 326/1261 [01:49<04:06,  3.80it/s][A
     26%|██▌       | 327/1261 [01:50<04:15,  3.66it/s][A
     26%|██▌       | 328/1261 [01:50<04:09,  3.74it/s][A
     26%|██▌       | 329/1261 [01:50<04:29,  3.46it/s][A
     26%|██▌       | 330/1261 [01:51<04:44,  3.28it/s][A
     26%|██▌       | 331/1261 [01:51<04:48,  3.23it/s][A
     26%|██▋       | 332/1261 [01:51<04:52,  3.17it/s][A
     26%|██▋       | 333/1261 [01:51<04:35,  3.36it/s][A
     26%|██▋       | 334/1261 [01:52<04:17,  3.60it/s][A
     27%|██▋       | 335/1261 [01:52<04:15,  3.62it/s][A
     27%|██▋       | 336/1261 [01:52<04:05,  3.76it/s][A
     27%|██▋       | 337/1261 [01:52<04:09,  3.71it/s][A
     27%|██▋       | 338/1261 [01:53<04:09,  3.70it/s][A
     27%|██▋       | 339/1261 [01:53<04:12,  3.65it/s][A
     27%|██▋       | 340/1261 [01:53<04:10,  3.68it/s][A
     27%|██▋       | 341/1261 [01:54<04:07,  3.72it/s][A
     27%|██▋       | 342/1261 [01:54<03:56,  3.89it/s][A
     27%|██▋       | 343/1261 [01:54<04:28,  3.42it/s][A
     27%|██▋       | 344/1261 [01:55<04:45,  3.21it/s][A
     27%|██▋       | 345/1261 [01:55<04:47,  3.19it/s][A
     27%|██▋       | 346/1261 [01:55<04:58,  3.07it/s][A
     28%|██▊       | 347/1261 [01:55<04:48,  3.17it/s][A
     28%|██▊       | 348/1261 [01:56<04:53,  3.11it/s][A
     28%|██▊       | 349/1261 [01:56<04:38,  3.27it/s][A
     28%|██▊       | 350/1261 [01:56<04:30,  3.37it/s][A
     28%|██▊       | 351/1261 [01:57<04:26,  3.42it/s][A
     28%|██▊       | 352/1261 [01:57<04:17,  3.53it/s][A
     28%|██▊       | 353/1261 [01:57<04:20,  3.49it/s][A
     28%|██▊       | 354/1261 [01:57<04:16,  3.54it/s][A
     28%|██▊       | 355/1261 [01:58<04:21,  3.46it/s][A
     28%|██▊       | 356/1261 [01:58<04:21,  3.46it/s][A
     28%|██▊       | 357/1261 [01:58<04:09,  3.62it/s][A
     28%|██▊       | 358/1261 [01:59<04:00,  3.75it/s][A
     28%|██▊       | 359/1261 [01:59<03:53,  3.86it/s][A
     29%|██▊       | 360/1261 [01:59<03:40,  4.09it/s][A
     29%|██▊       | 361/1261 [01:59<03:43,  4.03it/s][A
     29%|██▊       | 362/1261 [02:00<03:38,  4.11it/s][A
     29%|██▉       | 363/1261 [02:00<03:50,  3.90it/s][A
     29%|██▉       | 364/1261 [02:00<04:00,  3.73it/s][A
     29%|██▉       | 365/1261 [02:00<04:05,  3.66it/s][A
     29%|██▉       | 366/1261 [02:01<04:02,  3.69it/s][A
     29%|██▉       | 367/1261 [02:01<04:03,  3.68it/s][A
     29%|██▉       | 368/1261 [02:01<04:02,  3.68it/s][A
     29%|██▉       | 369/1261 [02:01<03:56,  3.78it/s][A
     29%|██▉       | 370/1261 [02:02<03:53,  3.82it/s][A
     29%|██▉       | 371/1261 [02:02<03:51,  3.84it/s][A
     30%|██▉       | 372/1261 [02:02<03:58,  3.73it/s][A
     30%|██▉       | 373/1261 [02:02<03:58,  3.73it/s][A
     30%|██▉       | 374/1261 [02:03<03:46,  3.91it/s][A
     30%|██▉       | 375/1261 [02:03<03:48,  3.87it/s][A
     30%|██▉       | 376/1261 [02:03<03:48,  3.88it/s][A
     30%|██▉       | 377/1261 [02:04<03:57,  3.73it/s][A
     30%|██▉       | 378/1261 [02:04<03:46,  3.89it/s][A
     30%|███       | 379/1261 [02:04<03:48,  3.86it/s][A
     30%|███       | 380/1261 [02:04<03:41,  3.97it/s][A
     30%|███       | 381/1261 [02:05<03:52,  3.78it/s][A
     30%|███       | 382/1261 [02:05<03:50,  3.81it/s][A
     30%|███       | 383/1261 [02:05<03:44,  3.91it/s][A
     30%|███       | 384/1261 [02:05<03:47,  3.86it/s][A
     31%|███       | 385/1261 [02:06<03:49,  3.81it/s][A
     31%|███       | 386/1261 [02:06<03:47,  3.85it/s][A
     31%|███       | 387/1261 [02:06<03:47,  3.84it/s][A
     31%|███       | 388/1261 [02:06<03:47,  3.83it/s][A
     31%|███       | 389/1261 [02:07<03:49,  3.80it/s][A
     31%|███       | 390/1261 [02:07<03:50,  3.78it/s][A
     31%|███       | 391/1261 [02:07<03:51,  3.76it/s][A
     31%|███       | 392/1261 [02:07<03:59,  3.62it/s][A
     31%|███       | 393/1261 [02:08<03:53,  3.71it/s][A
     31%|███       | 394/1261 [02:08<03:55,  3.68it/s][A
     31%|███▏      | 395/1261 [02:08<03:58,  3.63it/s][A
     31%|███▏      | 396/1261 [02:09<03:56,  3.66it/s][A
     31%|███▏      | 397/1261 [02:09<03:55,  3.67it/s][A
     32%|███▏      | 398/1261 [02:09<03:53,  3.69it/s][A
     32%|███▏      | 399/1261 [02:09<03:56,  3.65it/s][A
     32%|███▏      | 400/1261 [02:10<03:58,  3.61it/s][A
     32%|███▏      | 401/1261 [02:10<03:59,  3.59it/s][A
     32%|███▏      | 402/1261 [02:10<03:57,  3.62it/s][A
     32%|███▏      | 403/1261 [02:10<03:56,  3.63it/s][A
     32%|███▏      | 404/1261 [02:11<04:03,  3.53it/s][A
     32%|███▏      | 405/1261 [02:11<03:57,  3.61it/s][A
     32%|███▏      | 406/1261 [02:11<04:14,  3.36it/s][A
     32%|███▏      | 407/1261 [02:12<04:12,  3.38it/s][A
     32%|███▏      | 408/1261 [02:12<04:04,  3.49it/s][A
     32%|███▏      | 409/1261 [02:12<04:03,  3.50it/s][A
     33%|███▎      | 410/1261 [02:13<04:02,  3.51it/s][A
     33%|███▎      | 411/1261 [02:13<04:07,  3.44it/s][A
     33%|███▎      | 412/1261 [02:13<04:09,  3.41it/s][A
     33%|███▎      | 413/1261 [02:13<04:03,  3.49it/s][A
     33%|███▎      | 414/1261 [02:14<03:43,  3.78it/s][A
     33%|███▎      | 415/1261 [02:14<03:41,  3.81it/s][A
     33%|███▎      | 416/1261 [02:14<03:43,  3.79it/s][A
     33%|███▎      | 417/1261 [02:14<03:37,  3.88it/s][A
     33%|███▎      | 418/1261 [02:15<03:45,  3.73it/s][A
     33%|███▎      | 419/1261 [02:15<03:40,  3.82it/s][A
     33%|███▎      | 420/1261 [02:15<03:46,  3.71it/s][A
     33%|███▎      | 421/1261 [02:15<03:45,  3.73it/s][A
     33%|███▎      | 422/1261 [02:16<03:41,  3.79it/s][A
     34%|███▎      | 423/1261 [02:16<03:40,  3.80it/s][A
     34%|███▎      | 424/1261 [02:16<03:45,  3.71it/s][A
     34%|███▎      | 425/1261 [02:17<03:44,  3.73it/s][A
     34%|███▍      | 426/1261 [02:17<03:45,  3.70it/s][A
     34%|███▍      | 427/1261 [02:17<03:48,  3.65it/s][A
     34%|███▍      | 428/1261 [02:17<03:51,  3.59it/s][A
     34%|███▍      | 429/1261 [02:18<03:50,  3.61it/s][A
     34%|███▍      | 430/1261 [02:18<03:51,  3.59it/s][A
     34%|███▍      | 431/1261 [02:18<03:58,  3.48it/s][A
     34%|███▍      | 432/1261 [02:19<03:57,  3.49it/s][A
     34%|███▍      | 433/1261 [02:19<03:55,  3.51it/s][A
     34%|███▍      | 434/1261 [02:19<03:50,  3.58it/s][A
     34%|███▍      | 435/1261 [02:19<03:53,  3.54it/s][A
     35%|███▍      | 436/1261 [02:20<03:47,  3.63it/s][A
     35%|███▍      | 437/1261 [02:20<03:45,  3.65it/s][A
     35%|███▍      | 438/1261 [02:20<03:39,  3.74it/s][A
     35%|███▍      | 439/1261 [02:20<03:32,  3.86it/s][A
     35%|███▍      | 440/1261 [02:21<03:27,  3.95it/s][A
     35%|███▍      | 441/1261 [02:21<03:36,  3.79it/s][A
     35%|███▌      | 442/1261 [02:21<03:35,  3.81it/s][A
     35%|███▌      | 443/1261 [02:21<03:45,  3.63it/s][A
     35%|███▌      | 444/1261 [02:22<03:47,  3.59it/s][A
     35%|███▌      | 445/1261 [02:22<03:52,  3.52it/s][A
     35%|███▌      | 446/1261 [02:22<03:44,  3.63it/s][A
     35%|███▌      | 447/1261 [02:23<03:38,  3.73it/s][A
     36%|███▌      | 448/1261 [02:23<03:34,  3.78it/s][A
     36%|███▌      | 449/1261 [02:23<03:37,  3.73it/s][A
     36%|███▌      | 450/1261 [02:23<03:35,  3.76it/s][A
     36%|███▌      | 451/1261 [02:24<03:38,  3.71it/s][A
     36%|███▌      | 452/1261 [02:24<03:43,  3.62it/s][A
     36%|███▌      | 453/1261 [02:24<03:42,  3.62it/s][A
     36%|███▌      | 454/1261 [02:24<03:34,  3.77it/s][A
     36%|███▌      | 455/1261 [02:25<03:38,  3.69it/s][A
     36%|███▌      | 456/1261 [02:25<03:36,  3.72it/s][A
     36%|███▌      | 457/1261 [02:25<03:28,  3.86it/s][A
     36%|███▋      | 458/1261 [02:25<03:24,  3.94it/s][A
     36%|███▋      | 459/1261 [02:26<03:29,  3.82it/s][A
     36%|███▋      | 460/1261 [02:26<03:31,  3.79it/s][A
     37%|███▋      | 461/1261 [02:26<03:34,  3.73it/s][A
     37%|███▋      | 462/1261 [02:27<03:36,  3.70it/s][A
     37%|███▋      | 463/1261 [02:27<03:41,  3.60it/s][A
     37%|███▋      | 464/1261 [02:27<03:42,  3.58it/s][A
     37%|███▋      | 465/1261 [02:27<03:55,  3.38it/s][A
     37%|███▋      | 466/1261 [02:28<03:56,  3.36it/s][A
     37%|███▋      | 467/1261 [02:28<04:12,  3.15it/s][A
     37%|███▋      | 468/1261 [02:28<04:04,  3.24it/s][A
     37%|███▋      | 469/1261 [02:29<03:49,  3.45it/s][A
     37%|███▋      | 470/1261 [02:29<03:43,  3.54it/s][A
     37%|███▋      | 471/1261 [02:29<03:41,  3.57it/s][A
     37%|███▋      | 472/1261 [02:30<03:43,  3.53it/s][A
     38%|███▊      | 473/1261 [02:30<03:42,  3.53it/s][A
     38%|███▊      | 474/1261 [02:30<03:51,  3.40it/s][A
     38%|███▊      | 475/1261 [02:30<03:52,  3.39it/s][A
     38%|███▊      | 476/1261 [02:31<03:51,  3.40it/s][A
     38%|███▊      | 477/1261 [02:31<03:42,  3.53it/s][A
     38%|███▊      | 478/1261 [02:31<03:33,  3.66it/s][A
     38%|███▊      | 479/1261 [02:31<03:31,  3.70it/s][A
     38%|███▊      | 480/1261 [02:32<03:30,  3.71it/s][A
     38%|███▊      | 481/1261 [02:32<03:29,  3.72it/s][A
     38%|███▊      | 482/1261 [02:32<03:22,  3.84it/s][A
     38%|███▊      | 483/1261 [02:33<03:27,  3.75it/s][A
     38%|███▊      | 484/1261 [02:33<03:40,  3.52it/s][A
     38%|███▊      | 485/1261 [02:33<03:43,  3.48it/s][A
     39%|███▊      | 486/1261 [02:33<03:38,  3.54it/s][A
     39%|███▊      | 487/1261 [02:34<03:38,  3.53it/s][A
     39%|███▊      | 488/1261 [02:34<03:32,  3.64it/s][A
     39%|███▉      | 489/1261 [02:34<03:31,  3.64it/s][A
     39%|███▉      | 490/1261 [02:35<03:37,  3.55it/s][A
     39%|███▉      | 491/1261 [02:35<03:33,  3.60it/s][A
     39%|███▉      | 492/1261 [02:35<03:29,  3.66it/s][A
     39%|███▉      | 493/1261 [02:35<03:32,  3.62it/s][A
     39%|███▉      | 494/1261 [02:36<03:31,  3.62it/s][A
     39%|███▉      | 495/1261 [02:36<03:25,  3.73it/s][A
     39%|███▉      | 496/1261 [02:36<03:29,  3.66it/s][A
     39%|███▉      | 497/1261 [02:36<03:31,  3.62it/s][A
     39%|███▉      | 498/1261 [02:37<03:21,  3.79it/s][A
     40%|███▉      | 499/1261 [02:37<03:17,  3.85it/s][A
     40%|███▉      | 500/1261 [02:37<03:14,  3.91it/s][A
     40%|███▉      | 501/1261 [02:37<03:17,  3.85it/s][A
     40%|███▉      | 502/1261 [02:38<03:11,  3.95it/s][A
     40%|███▉      | 503/1261 [02:38<03:23,  3.72it/s][A
     40%|███▉      | 504/1261 [02:38<03:16,  3.86it/s][A
     40%|████      | 505/1261 [02:39<03:23,  3.72it/s][A
     40%|████      | 506/1261 [02:39<03:27,  3.63it/s][A
     40%|████      | 507/1261 [02:39<03:30,  3.58it/s][A
     40%|████      | 508/1261 [02:39<03:28,  3.61it/s][A
     40%|████      | 509/1261 [02:40<03:27,  3.63it/s][A
     40%|████      | 510/1261 [02:40<03:18,  3.79it/s][A
     41%|████      | 511/1261 [02:40<03:18,  3.78it/s][A
     41%|████      | 512/1261 [02:40<03:17,  3.79it/s][A
     41%|████      | 513/1261 [02:41<03:19,  3.75it/s][A
     41%|████      | 514/1261 [02:41<03:19,  3.74it/s][A
     41%|████      | 515/1261 [02:41<03:33,  3.49it/s][A
     41%|████      | 516/1261 [02:42<03:31,  3.53it/s][A
     41%|████      | 517/1261 [02:42<03:29,  3.55it/s][A
     41%|████      | 518/1261 [02:42<03:28,  3.56it/s][A
     41%|████      | 519/1261 [02:42<03:32,  3.49it/s][A
     41%|████      | 520/1261 [02:43<03:19,  3.71it/s][A
     41%|████▏     | 521/1261 [02:43<03:22,  3.66it/s][A
     41%|████▏     | 522/1261 [02:43<03:16,  3.75it/s][A
     41%|████▏     | 523/1261 [02:43<03:20,  3.67it/s][A
     42%|████▏     | 524/1261 [02:44<03:30,  3.50it/s][A
     42%|████▏     | 525/1261 [02:44<03:32,  3.46it/s][A
     42%|████▏     | 526/1261 [02:44<03:26,  3.56it/s][A
     42%|████▏     | 527/1261 [02:45<03:24,  3.59it/s][A
     42%|████▏     | 528/1261 [02:45<03:20,  3.65it/s][A
     42%|████▏     | 529/1261 [02:45<03:22,  3.62it/s][A
     42%|████▏     | 530/1261 [02:45<03:17,  3.69it/s][A
     42%|████▏     | 531/1261 [02:46<03:17,  3.70it/s][A
     42%|████▏     | 532/1261 [02:46<03:15,  3.74it/s][A
     42%|████▏     | 533/1261 [02:46<03:24,  3.57it/s][A
     42%|████▏     | 534/1261 [02:47<03:34,  3.40it/s][A
     42%|████▏     | 535/1261 [02:47<03:42,  3.27it/s][A
     43%|████▎     | 536/1261 [02:47<03:25,  3.53it/s][A
     43%|████▎     | 537/1261 [02:47<03:09,  3.81it/s][A
     43%|████▎     | 538/1261 [02:48<03:11,  3.78it/s][A
     43%|████▎     | 539/1261 [02:48<03:12,  3.76it/s][A
     43%|████▎     | 540/1261 [02:48<03:16,  3.67it/s][A
     43%|████▎     | 541/1261 [02:48<03:11,  3.77it/s][A
     43%|████▎     | 542/1261 [02:49<03:01,  3.95it/s][A
     43%|████▎     | 543/1261 [02:49<03:06,  3.84it/s][A
     43%|████▎     | 544/1261 [02:49<03:08,  3.81it/s][A
     43%|████▎     | 545/1261 [02:49<03:00,  3.97it/s][A
     43%|████▎     | 546/1261 [02:50<02:51,  4.17it/s][A
     43%|████▎     | 547/1261 [02:50<02:52,  4.15it/s][A
     43%|████▎     | 548/1261 [02:50<02:55,  4.07it/s][A
     44%|████▎     | 549/1261 [02:50<02:56,  4.03it/s][A
     44%|████▎     | 550/1261 [02:51<02:56,  4.04it/s][A
     44%|████▎     | 551/1261 [02:51<02:52,  4.12it/s][A
     44%|████▍     | 552/1261 [02:51<02:58,  3.97it/s][A
     44%|████▍     | 553/1261 [02:51<02:52,  4.11it/s][A
     44%|████▍     | 554/1261 [02:52<02:56,  4.01it/s][A
     44%|████▍     | 555/1261 [02:52<03:00,  3.92it/s][A
     44%|████▍     | 556/1261 [02:52<03:09,  3.73it/s][A
     44%|████▍     | 557/1261 [02:52<03:11,  3.67it/s][A
     44%|████▍     | 558/1261 [02:53<03:02,  3.84it/s][A
     44%|████▍     | 559/1261 [02:53<03:12,  3.65it/s][A
     44%|████▍     | 560/1261 [02:53<03:28,  3.37it/s][A
     44%|████▍     | 561/1261 [02:54<03:36,  3.23it/s][A
     45%|████▍     | 562/1261 [02:54<03:45,  3.11it/s][A
     45%|████▍     | 563/1261 [02:54<03:56,  2.95it/s][A
     45%|████▍     | 564/1261 [02:55<04:04,  2.85it/s][A
     45%|████▍     | 565/1261 [02:55<03:56,  2.95it/s][A
     45%|████▍     | 566/1261 [02:55<03:58,  2.91it/s][A
     45%|████▍     | 567/1261 [02:56<03:56,  2.94it/s][A
     45%|████▌     | 568/1261 [02:56<03:55,  2.94it/s][A
     45%|████▌     | 569/1261 [02:57<04:00,  2.88it/s][A
     45%|████▌     | 570/1261 [02:57<03:56,  2.92it/s][A
     45%|████▌     | 571/1261 [02:57<03:58,  2.89it/s][A
     45%|████▌     | 572/1261 [02:58<04:01,  2.85it/s][A
     45%|████▌     | 573/1261 [02:58<04:28,  2.56it/s][A
     46%|████▌     | 574/1261 [02:58<04:31,  2.53it/s][A
     46%|████▌     | 575/1261 [02:59<04:29,  2.55it/s][A
     46%|████▌     | 576/1261 [02:59<04:23,  2.60it/s][A
     46%|████▌     | 577/1261 [03:00<04:11,  2.72it/s][A
     46%|████▌     | 578/1261 [03:00<04:17,  2.65it/s][A
     46%|████▌     | 579/1261 [03:00<04:18,  2.64it/s][A
     46%|████▌     | 580/1261 [03:01<04:10,  2.72it/s][A
     46%|████▌     | 581/1261 [03:01<04:20,  2.61it/s][A
     46%|████▌     | 582/1261 [03:01<04:16,  2.64it/s][A
     46%|████▌     | 583/1261 [03:02<04:36,  2.45it/s][A
     46%|████▋     | 584/1261 [03:02<04:24,  2.56it/s][A
     46%|████▋     | 585/1261 [03:03<04:31,  2.49it/s][A
     46%|████▋     | 586/1261 [03:03<04:20,  2.59it/s][A
     47%|████▋     | 587/1261 [03:03<04:12,  2.67it/s][A
     47%|████▋     | 588/1261 [03:04<04:01,  2.78it/s][A
     47%|████▋     | 589/1261 [03:04<04:13,  2.65it/s][A
     47%|████▋     | 590/1261 [03:04<04:03,  2.75it/s][A
     47%|████▋     | 591/1261 [03:05<03:51,  2.89it/s][A
     47%|████▋     | 592/1261 [03:05<03:54,  2.85it/s][A
     47%|████▋     | 593/1261 [03:05<03:52,  2.87it/s][A
     47%|████▋     | 594/1261 [03:06<03:53,  2.86it/s][A
     47%|████▋     | 595/1261 [03:06<03:45,  2.96it/s][A
     47%|████▋     | 596/1261 [03:07<03:57,  2.80it/s][A
     47%|████▋     | 597/1261 [03:07<03:46,  2.93it/s][A
     47%|████▋     | 598/1261 [03:07<03:49,  2.89it/s][A
     48%|████▊     | 599/1261 [03:08<03:56,  2.80it/s][A
     48%|████▊     | 600/1261 [03:08<03:50,  2.87it/s][A
     48%|████▊     | 601/1261 [03:08<03:59,  2.75it/s][A
     48%|████▊     | 602/1261 [03:09<03:56,  2.78it/s][A
     48%|████▊     | 603/1261 [03:09<03:52,  2.83it/s][A
     48%|████▊     | 604/1261 [03:09<03:56,  2.77it/s][A
     48%|████▊     | 605/1261 [03:10<04:03,  2.69it/s][A
     48%|████▊     | 606/1261 [03:10<04:06,  2.65it/s][A
     48%|████▊     | 607/1261 [03:11<03:57,  2.75it/s][A
     48%|████▊     | 608/1261 [03:11<03:47,  2.87it/s][A
     48%|████▊     | 609/1261 [03:11<03:55,  2.77it/s][A
     48%|████▊     | 610/1261 [03:12<03:53,  2.79it/s][A
     48%|████▊     | 611/1261 [03:12<03:51,  2.81it/s][A
     49%|████▊     | 612/1261 [03:12<04:13,  2.56it/s][A
     49%|████▊     | 613/1261 [03:13<04:10,  2.59it/s][A
     49%|████▊     | 614/1261 [03:13<03:52,  2.79it/s][A
     49%|████▉     | 615/1261 [03:13<03:43,  2.89it/s][A
     49%|████▉     | 616/1261 [03:14<03:46,  2.85it/s][A
     49%|████▉     | 617/1261 [03:14<03:47,  2.83it/s][A
     49%|████▉     | 618/1261 [03:14<03:43,  2.88it/s][A
     49%|████▉     | 619/1261 [03:15<03:30,  3.04it/s][A
     49%|████▉     | 620/1261 [03:15<03:47,  2.82it/s][A
     49%|████▉     | 621/1261 [03:15<03:46,  2.83it/s][A
     49%|████▉     | 622/1261 [03:16<03:56,  2.71it/s][A
     49%|████▉     | 623/1261 [03:16<03:53,  2.73it/s][A
     49%|████▉     | 624/1261 [03:17<03:44,  2.84it/s][A
     50%|████▉     | 625/1261 [03:17<03:45,  2.82it/s][A
     50%|████▉     | 626/1261 [03:17<03:37,  2.92it/s][A
     50%|████▉     | 627/1261 [03:18<03:42,  2.85it/s][A
     50%|████▉     | 628/1261 [03:18<03:47,  2.78it/s][A
     50%|████▉     | 629/1261 [03:18<03:52,  2.72it/s][A
     50%|████▉     | 630/1261 [03:19<03:50,  2.73it/s][A
     50%|█████     | 631/1261 [03:19<03:43,  2.82it/s][A
     50%|█████     | 632/1261 [03:19<03:52,  2.71it/s][A
     50%|█████     | 633/1261 [03:20<03:56,  2.66it/s][A
     50%|█████     | 634/1261 [03:20<04:04,  2.56it/s][A
     50%|█████     | 635/1261 [03:21<03:56,  2.65it/s][A
     50%|█████     | 636/1261 [03:21<03:48,  2.74it/s][A
     51%|█████     | 637/1261 [03:21<03:40,  2.83it/s][A
     51%|█████     | 638/1261 [03:22<03:43,  2.79it/s][A
     51%|█████     | 639/1261 [03:22<03:46,  2.74it/s][A
     51%|█████     | 640/1261 [03:22<03:47,  2.73it/s][A
     51%|█████     | 641/1261 [03:23<03:32,  2.91it/s][A
     51%|█████     | 642/1261 [03:23<03:31,  2.93it/s][A
     51%|█████     | 643/1261 [03:23<03:33,  2.90it/s][A
     51%|█████     | 644/1261 [03:24<03:25,  3.01it/s][A
     51%|█████     | 645/1261 [03:24<03:25,  3.00it/s][A
     51%|█████     | 646/1261 [03:24<03:29,  2.94it/s][A
     51%|█████▏    | 647/1261 [03:25<03:33,  2.87it/s][A
     51%|█████▏    | 648/1261 [03:25<03:24,  2.99it/s][A
     51%|█████▏    | 649/1261 [03:25<03:34,  2.85it/s][A
     52%|█████▏    | 650/1261 [03:26<03:29,  2.91it/s][A
     52%|█████▏    | 651/1261 [03:26<03:22,  3.02it/s][A
     52%|█████▏    | 652/1261 [03:26<03:19,  3.06it/s][A
     52%|█████▏    | 653/1261 [03:27<03:38,  2.78it/s][A
     52%|█████▏    | 654/1261 [03:27<03:29,  2.90it/s][A
     52%|█████▏    | 655/1261 [03:27<03:23,  2.98it/s][A
     52%|█████▏    | 656/1261 [03:28<03:21,  3.00it/s][A
     52%|█████▏    | 657/1261 [03:28<03:32,  2.84it/s][A
     52%|█████▏    | 658/1261 [03:29<03:31,  2.85it/s][A
     52%|█████▏    | 659/1261 [03:29<03:34,  2.81it/s][A
     52%|█████▏    | 660/1261 [03:29<03:28,  2.88it/s][A
     52%|█████▏    | 661/1261 [03:30<03:36,  2.78it/s][A
     52%|█████▏    | 662/1261 [03:30<03:29,  2.86it/s][A
     53%|█████▎    | 663/1261 [03:30<03:18,  3.01it/s][A
     53%|█████▎    | 664/1261 [03:31<03:18,  3.01it/s][A
     53%|█████▎    | 665/1261 [03:31<03:31,  2.81it/s][A
     53%|█████▎    | 666/1261 [03:31<03:50,  2.58it/s][A
     53%|█████▎    | 667/1261 [03:32<03:46,  2.62it/s][A
     53%|█████▎    | 668/1261 [03:32<03:47,  2.61it/s][A
     53%|█████▎    | 669/1261 [03:33<03:51,  2.56it/s][A
     53%|█████▎    | 670/1261 [03:33<03:50,  2.57it/s][A
     53%|█████▎    | 671/1261 [03:33<03:45,  2.62it/s][A
     53%|█████▎    | 672/1261 [03:34<03:48,  2.58it/s][A
     53%|█████▎    | 673/1261 [03:34<03:47,  2.59it/s][A
     53%|█████▎    | 674/1261 [03:34<03:37,  2.69it/s][A
     54%|█████▎    | 675/1261 [03:35<03:48,  2.56it/s][A
     54%|█████▎    | 676/1261 [03:35<03:46,  2.59it/s][A
     54%|█████▎    | 677/1261 [03:36<03:49,  2.54it/s][A
     54%|█████▍    | 678/1261 [03:36<03:33,  2.74it/s][A
     54%|█████▍    | 679/1261 [03:36<03:26,  2.82it/s][A
     54%|█████▍    | 680/1261 [03:37<03:27,  2.81it/s][A
     54%|█████▍    | 681/1261 [03:37<03:27,  2.80it/s][A
     54%|█████▍    | 682/1261 [03:37<03:22,  2.86it/s][A
     54%|█████▍    | 683/1261 [03:38<03:27,  2.79it/s][A
     54%|█████▍    | 684/1261 [03:38<03:24,  2.82it/s][A
     54%|█████▍    | 685/1261 [03:38<03:25,  2.81it/s][A
     54%|█████▍    | 686/1261 [03:39<03:18,  2.90it/s][A
     54%|█████▍    | 687/1261 [03:39<03:05,  3.10it/s][A
     55%|█████▍    | 688/1261 [03:39<03:04,  3.11it/s][A
     55%|█████▍    | 689/1261 [03:40<03:08,  3.03it/s][A
     55%|█████▍    | 690/1261 [03:40<03:13,  2.95it/s][A
     55%|█████▍    | 691/1261 [03:40<03:09,  3.01it/s][A
     55%|█████▍    | 692/1261 [03:41<03:08,  3.01it/s][A
     55%|█████▍    | 693/1261 [03:41<03:17,  2.87it/s][A
     55%|█████▌    | 694/1261 [03:41<03:15,  2.90it/s][A
     55%|█████▌    | 695/1261 [03:42<03:10,  2.98it/s][A
     55%|█████▌    | 696/1261 [03:42<03:00,  3.14it/s][A
     55%|█████▌    | 697/1261 [03:42<02:51,  3.29it/s][A
     55%|█████▌    | 698/1261 [03:43<02:52,  3.25it/s][A
     55%|█████▌    | 699/1261 [03:43<03:05,  3.03it/s][A
     56%|█████▌    | 700/1261 [03:43<03:11,  2.93it/s][A
     56%|█████▌    | 701/1261 [03:44<03:12,  2.90it/s][A
     56%|█████▌    | 702/1261 [03:44<03:27,  2.70it/s][A
     56%|█████▌    | 703/1261 [03:44<03:19,  2.80it/s][A
     56%|█████▌    | 704/1261 [03:45<03:20,  2.77it/s][A
     56%|█████▌    | 705/1261 [03:45<03:14,  2.86it/s][A
     56%|█████▌    | 706/1261 [03:45<02:59,  3.10it/s][A
     56%|█████▌    | 707/1261 [03:46<03:03,  3.01it/s][A
     56%|█████▌    | 708/1261 [03:46<03:03,  3.02it/s][A
     56%|█████▌    | 709/1261 [03:46<03:10,  2.90it/s][A
     56%|█████▋    | 710/1261 [03:47<03:14,  2.84it/s][A
     56%|█████▋    | 711/1261 [03:47<03:18,  2.77it/s][A
     56%|█████▋    | 712/1261 [03:48<03:14,  2.83it/s][A
     57%|█████▋    | 713/1261 [03:48<03:02,  3.00it/s][A
     57%|█████▋    | 714/1261 [03:48<03:04,  2.96it/s][A
     57%|█████▋    | 715/1261 [03:49<02:57,  3.07it/s][A
     57%|█████▋    | 716/1261 [03:49<02:57,  3.08it/s][A
     57%|█████▋    | 717/1261 [03:49<02:52,  3.15it/s][A
     57%|█████▋    | 718/1261 [03:49<02:58,  3.05it/s][A
     57%|█████▋    | 719/1261 [03:50<03:06,  2.90it/s][A
     57%|█████▋    | 720/1261 [03:50<03:16,  2.76it/s][A
     57%|█████▋    | 721/1261 [03:51<03:02,  2.95it/s][A
     57%|█████▋    | 722/1261 [03:51<03:00,  2.98it/s][A
     57%|█████▋    | 723/1261 [03:51<03:00,  2.98it/s][A
     57%|█████▋    | 724/1261 [03:52<02:57,  3.03it/s][A
     57%|█████▋    | 725/1261 [03:52<02:53,  3.10it/s][A
     58%|█████▊    | 726/1261 [03:52<02:58,  2.99it/s][A
     58%|█████▊    | 727/1261 [03:53<03:06,  2.87it/s][A
     58%|█████▊    | 728/1261 [03:53<03:02,  2.91it/s][A
     58%|█████▊    | 729/1261 [03:53<03:05,  2.87it/s][A
     58%|█████▊    | 730/1261 [03:54<03:01,  2.92it/s][A
     58%|█████▊    | 731/1261 [03:54<03:07,  2.83it/s][A
     58%|█████▊    | 732/1261 [03:54<03:07,  2.82it/s][A
     58%|█████▊    | 733/1261 [03:55<03:15,  2.70it/s][A
     58%|█████▊    | 734/1261 [03:55<03:15,  2.70it/s][A
     58%|█████▊    | 735/1261 [03:55<03:11,  2.75it/s][A
     58%|█████▊    | 736/1261 [03:56<03:02,  2.87it/s][A
     58%|█████▊    | 737/1261 [03:56<02:52,  3.04it/s][A
     59%|█████▊    | 738/1261 [03:56<02:51,  3.04it/s][A
     59%|█████▊    | 739/1261 [03:57<02:59,  2.92it/s][A
     59%|█████▊    | 740/1261 [03:57<03:08,  2.77it/s][A
     59%|█████▉    | 741/1261 [03:58<03:13,  2.68it/s][A
     59%|█████▉    | 742/1261 [03:58<03:00,  2.88it/s][A
     59%|█████▉    | 743/1261 [03:58<02:51,  3.02it/s][A
     59%|█████▉    | 744/1261 [03:58<02:47,  3.09it/s][A
     59%|█████▉    | 745/1261 [03:59<02:45,  3.11it/s][A
     59%|█████▉    | 746/1261 [03:59<02:46,  3.09it/s][A
     59%|█████▉    | 747/1261 [03:59<02:49,  3.03it/s][A
     59%|█████▉    | 748/1261 [04:00<02:58,  2.88it/s][A
     59%|█████▉    | 749/1261 [04:00<03:01,  2.83it/s][A
     59%|█████▉    | 750/1261 [04:01<03:01,  2.82it/s][A
     60%|█████▉    | 751/1261 [04:01<03:02,  2.79it/s][A
     60%|█████▉    | 752/1261 [04:01<02:56,  2.88it/s][A
     60%|█████▉    | 753/1261 [04:02<03:04,  2.75it/s][A
     60%|█████▉    | 754/1261 [04:02<02:58,  2.84it/s][A
     60%|█████▉    | 755/1261 [04:02<02:58,  2.83it/s][A
     60%|█████▉    | 756/1261 [04:03<02:55,  2.88it/s][A
     60%|██████    | 757/1261 [04:03<02:54,  2.89it/s][A
     60%|██████    | 758/1261 [04:03<02:57,  2.83it/s][A
     60%|██████    | 759/1261 [04:04<02:57,  2.83it/s][A
     60%|██████    | 760/1261 [04:04<02:51,  2.92it/s][A
     60%|██████    | 761/1261 [04:04<02:47,  2.98it/s][A
     60%|██████    | 762/1261 [04:05<02:56,  2.83it/s][A
     61%|██████    | 763/1261 [04:05<02:49,  2.94it/s][A
     61%|██████    | 764/1261 [04:05<02:52,  2.89it/s][A
     61%|██████    | 765/1261 [04:06<02:53,  2.85it/s][A
     61%|██████    | 766/1261 [04:06<02:55,  2.83it/s][A
     61%|██████    | 767/1261 [04:07<03:06,  2.65it/s][A
     61%|██████    | 768/1261 [04:07<02:56,  2.79it/s][A
     61%|██████    | 769/1261 [04:07<02:59,  2.74it/s][A
     61%|██████    | 770/1261 [04:08<02:55,  2.81it/s][A
     61%|██████    | 771/1261 [04:08<02:49,  2.90it/s][A
     61%|██████    | 772/1261 [04:08<02:41,  3.03it/s][A
     61%|██████▏   | 773/1261 [04:09<02:50,  2.87it/s][A
     61%|██████▏   | 774/1261 [04:09<02:46,  2.92it/s][A
     61%|██████▏   | 775/1261 [04:09<02:48,  2.88it/s][A
     62%|██████▏   | 776/1261 [04:10<02:53,  2.80it/s][A
     62%|██████▏   | 777/1261 [04:10<02:52,  2.80it/s][A
     62%|██████▏   | 778/1261 [04:10<02:46,  2.89it/s][A
     62%|██████▏   | 779/1261 [04:11<02:37,  3.06it/s][A
     62%|██████▏   | 780/1261 [04:11<02:31,  3.17it/s][A
     62%|██████▏   | 781/1261 [04:11<02:33,  3.13it/s][A
     62%|██████▏   | 782/1261 [04:12<02:38,  3.03it/s][A
     62%|██████▏   | 783/1261 [04:12<02:33,  3.12it/s][A
     62%|██████▏   | 784/1261 [04:12<02:36,  3.06it/s][A
     62%|██████▏   | 785/1261 [04:13<02:39,  2.99it/s][A
     62%|██████▏   | 786/1261 [04:13<02:41,  2.93it/s][A
     62%|██████▏   | 787/1261 [04:13<02:33,  3.09it/s][A
     62%|██████▏   | 788/1261 [04:14<02:47,  2.82it/s][A
     63%|██████▎   | 789/1261 [04:14<02:38,  2.97it/s][A
     63%|██████▎   | 790/1261 [04:14<02:34,  3.04it/s][A
     63%|██████▎   | 791/1261 [04:15<02:46,  2.83it/s][A
     63%|██████▎   | 792/1261 [04:15<02:46,  2.82it/s][A
     63%|██████▎   | 793/1261 [04:15<02:45,  2.83it/s][A
     63%|██████▎   | 794/1261 [04:16<02:54,  2.68it/s][A
     63%|██████▎   | 795/1261 [04:16<02:51,  2.72it/s][A
     63%|██████▎   | 796/1261 [04:17<02:52,  2.69it/s][A
     63%|██████▎   | 797/1261 [04:17<02:46,  2.78it/s][A
     63%|██████▎   | 798/1261 [04:17<02:38,  2.92it/s][A
     63%|██████▎   | 799/1261 [04:18<02:37,  2.94it/s][A
     63%|██████▎   | 800/1261 [04:18<02:40,  2.87it/s][A
     64%|██████▎   | 801/1261 [04:18<02:38,  2.91it/s][A
     64%|██████▎   | 802/1261 [04:19<02:42,  2.83it/s][A
     64%|██████▎   | 803/1261 [04:19<02:35,  2.95it/s][A
     64%|██████▍   | 804/1261 [04:19<02:47,  2.73it/s][A
     64%|██████▍   | 805/1261 [04:20<02:47,  2.73it/s][A
     64%|██████▍   | 806/1261 [04:20<02:42,  2.80it/s][A
     64%|██████▍   | 807/1261 [04:20<02:40,  2.84it/s][A
     64%|██████▍   | 808/1261 [04:21<02:41,  2.80it/s][A
     64%|██████▍   | 809/1261 [04:21<02:39,  2.84it/s][A
     64%|██████▍   | 810/1261 [04:22<02:45,  2.72it/s][A
     64%|██████▍   | 811/1261 [04:22<02:34,  2.92it/s][A
     64%|██████▍   | 812/1261 [04:22<02:31,  2.97it/s][A
     64%|██████▍   | 813/1261 [04:22<02:30,  2.97it/s][A
     65%|██████▍   | 814/1261 [04:23<02:35,  2.88it/s][A
     65%|██████▍   | 815/1261 [04:23<02:31,  2.93it/s][A
     65%|██████▍   | 816/1261 [04:23<02:33,  2.90it/s][A
     65%|██████▍   | 817/1261 [04:24<02:30,  2.94it/s][A
     65%|██████▍   | 818/1261 [04:24<02:35,  2.85it/s][A
     65%|██████▍   | 819/1261 [04:25<02:38,  2.78it/s][A
     65%|██████▌   | 820/1261 [04:25<02:37,  2.81it/s][A
     65%|██████▌   | 821/1261 [04:25<02:28,  2.95it/s][A
     65%|██████▌   | 822/1261 [04:26<02:28,  2.97it/s][A
     65%|██████▌   | 823/1261 [04:26<02:30,  2.91it/s][A
     65%|██████▌   | 824/1261 [04:26<02:37,  2.78it/s][A
     65%|██████▌   | 825/1261 [04:27<02:35,  2.81it/s][A
     66%|██████▌   | 826/1261 [04:27<02:39,  2.73it/s][A
     66%|██████▌   | 827/1261 [04:27<02:38,  2.74it/s][A
     66%|██████▌   | 828/1261 [04:28<02:34,  2.81it/s][A
     66%|██████▌   | 829/1261 [04:28<02:42,  2.65it/s][A
     66%|██████▌   | 830/1261 [04:29<02:38,  2.73it/s][A
     66%|██████▌   | 831/1261 [04:29<02:31,  2.83it/s][A
     66%|██████▌   | 832/1261 [04:29<02:39,  2.69it/s][A
     66%|██████▌   | 833/1261 [04:30<02:36,  2.73it/s][A
     66%|██████▌   | 834/1261 [04:30<02:31,  2.81it/s][A
     66%|██████▌   | 835/1261 [04:30<02:27,  2.88it/s][A
     66%|██████▋   | 836/1261 [04:31<02:32,  2.78it/s][A
     66%|██████▋   | 837/1261 [04:31<02:37,  2.69it/s][A
     66%|██████▋   | 838/1261 [04:31<02:36,  2.71it/s][A
     67%|██████▋   | 839/1261 [04:32<02:33,  2.75it/s][A
     67%|██████▋   | 840/1261 [04:32<02:32,  2.76it/s][A
     67%|██████▋   | 841/1261 [04:33<02:35,  2.71it/s][A
     67%|██████▋   | 842/1261 [04:33<02:29,  2.81it/s][A
     67%|██████▋   | 843/1261 [04:33<02:29,  2.79it/s][A
     67%|██████▋   | 844/1261 [04:34<02:29,  2.78it/s][A
     67%|██████▋   | 845/1261 [04:34<02:34,  2.70it/s][A
     67%|██████▋   | 846/1261 [04:34<02:30,  2.76it/s][A
     67%|██████▋   | 847/1261 [04:35<02:30,  2.74it/s][A
     67%|██████▋   | 848/1261 [04:35<02:25,  2.84it/s][A
     67%|██████▋   | 849/1261 [04:35<02:17,  3.01it/s][A
     67%|██████▋   | 850/1261 [04:36<02:18,  2.98it/s][A
     67%|██████▋   | 851/1261 [04:36<02:20,  2.91it/s][A
     68%|██████▊   | 852/1261 [04:36<02:27,  2.77it/s][A
     68%|██████▊   | 853/1261 [04:37<02:20,  2.90it/s][A
     68%|██████▊   | 854/1261 [04:37<02:25,  2.80it/s][A
     68%|██████▊   | 855/1261 [04:37<02:25,  2.79it/s][A
     68%|██████▊   | 856/1261 [04:38<02:26,  2.77it/s][A
     68%|██████▊   | 857/1261 [04:38<02:29,  2.70it/s][A
     68%|██████▊   | 858/1261 [04:39<02:30,  2.67it/s][A
     68%|██████▊   | 859/1261 [04:39<02:36,  2.57it/s][A
     68%|██████▊   | 860/1261 [04:39<02:35,  2.58it/s][A
     68%|██████▊   | 861/1261 [04:40<02:37,  2.54it/s][A
     68%|██████▊   | 862/1261 [04:40<02:30,  2.64it/s][A
     68%|██████▊   | 863/1261 [04:41<02:28,  2.68it/s][A
     69%|██████▊   | 864/1261 [04:41<02:22,  2.78it/s][A
     69%|██████▊   | 865/1261 [04:41<02:15,  2.91it/s][A
     69%|██████▊   | 866/1261 [04:41<02:09,  3.05it/s][A
     69%|██████▉   | 867/1261 [04:42<02:18,  2.85it/s][A
     69%|██████▉   | 868/1261 [04:42<02:17,  2.85it/s][A
     69%|██████▉   | 869/1261 [04:43<02:22,  2.75it/s][A
     69%|██████▉   | 870/1261 [04:43<02:23,  2.73it/s][A
     69%|██████▉   | 871/1261 [04:43<02:14,  2.91it/s][A
     69%|██████▉   | 872/1261 [04:44<02:20,  2.78it/s][A
     69%|██████▉   | 873/1261 [04:44<02:16,  2.84it/s][A
     69%|██████▉   | 874/1261 [04:44<02:12,  2.92it/s][A
     69%|██████▉   | 875/1261 [04:45<02:13,  2.90it/s][A
     69%|██████▉   | 876/1261 [04:45<02:16,  2.82it/s][A
     70%|██████▉   | 877/1261 [04:45<02:20,  2.73it/s][A
     70%|██████▉   | 878/1261 [04:46<02:18,  2.77it/s][A
     70%|██████▉   | 879/1261 [04:46<02:15,  2.82it/s][A
     70%|██████▉   | 880/1261 [04:46<02:16,  2.80it/s][A
     70%|██████▉   | 881/1261 [04:47<02:14,  2.83it/s][A
     70%|██████▉   | 882/1261 [04:47<02:11,  2.89it/s][A
     70%|███████   | 883/1261 [04:48<02:12,  2.85it/s][A
     70%|███████   | 884/1261 [04:48<02:17,  2.73it/s][A
     70%|███████   | 885/1261 [04:48<02:20,  2.67it/s][A
     70%|███████   | 886/1261 [04:49<02:18,  2.71it/s][A
     70%|███████   | 887/1261 [04:49<02:18,  2.70it/s][A
     70%|███████   | 888/1261 [04:49<02:18,  2.68it/s][A
     70%|███████   | 889/1261 [04:50<02:21,  2.63it/s][A
     71%|███████   | 890/1261 [04:50<02:18,  2.67it/s][A
     71%|███████   | 891/1261 [04:50<02:12,  2.80it/s][A
     71%|███████   | 892/1261 [04:51<02:14,  2.74it/s][A
     71%|███████   | 893/1261 [04:51<02:12,  2.77it/s][A
     71%|███████   | 894/1261 [04:52<02:10,  2.82it/s][A
     71%|███████   | 895/1261 [04:52<02:10,  2.80it/s][A
     71%|███████   | 896/1261 [04:52<02:08,  2.83it/s][A
     71%|███████   | 897/1261 [04:53<01:58,  3.08it/s][A
     71%|███████   | 898/1261 [04:53<02:00,  3.02it/s][A
     71%|███████▏  | 899/1261 [04:53<01:55,  3.13it/s][A
     71%|███████▏  | 900/1261 [04:54<02:06,  2.86it/s][A
     71%|███████▏  | 901/1261 [04:54<02:03,  2.91it/s][A
     72%|███████▏  | 902/1261 [04:54<01:57,  3.05it/s][A
     72%|███████▏  | 903/1261 [04:55<02:01,  2.94it/s][A
     72%|███████▏  | 904/1261 [04:55<02:10,  2.74it/s][A
     72%|███████▏  | 905/1261 [04:55<02:06,  2.82it/s][A
     72%|███████▏  | 906/1261 [04:56<02:06,  2.80it/s][A
     72%|███████▏  | 907/1261 [04:56<02:02,  2.90it/s][A
     72%|███████▏  | 908/1261 [04:56<01:58,  2.97it/s][A
     72%|███████▏  | 909/1261 [04:57<01:56,  3.02it/s][A
     72%|███████▏  | 910/1261 [04:57<01:56,  3.01it/s][A
     72%|███████▏  | 911/1261 [04:57<01:57,  2.99it/s][A
     72%|███████▏  | 912/1261 [04:58<01:54,  3.05it/s][A
     72%|███████▏  | 913/1261 [04:58<01:56,  2.98it/s][A
     72%|███████▏  | 914/1261 [04:58<01:59,  2.90it/s][A
     73%|███████▎  | 915/1261 [04:59<01:57,  2.94it/s][A
     73%|███████▎  | 916/1261 [04:59<02:03,  2.80it/s][A
     73%|███████▎  | 917/1261 [04:59<02:00,  2.86it/s][A
     73%|███████▎  | 918/1261 [05:00<01:51,  3.08it/s][A
     73%|███████▎  | 919/1261 [05:00<01:50,  3.10it/s][A
     73%|███████▎  | 920/1261 [05:00<01:51,  3.06it/s][A
     73%|███████▎  | 921/1261 [05:01<01:56,  2.92it/s][A
     73%|███████▎  | 922/1261 [05:01<02:02,  2.78it/s][A
     73%|███████▎  | 923/1261 [05:01<02:04,  2.72it/s][A
     73%|███████▎  | 924/1261 [05:02<02:05,  2.70it/s][A
     73%|███████▎  | 925/1261 [05:02<02:06,  2.65it/s][A
     73%|███████▎  | 926/1261 [05:03<02:01,  2.75it/s][A
     74%|███████▎  | 927/1261 [05:03<01:56,  2.86it/s][A
     74%|███████▎  | 928/1261 [05:03<01:54,  2.92it/s][A
     74%|███████▎  | 929/1261 [05:04<01:48,  3.05it/s][A
     74%|███████▍  | 930/1261 [05:04<01:50,  3.00it/s][A
     74%|███████▍  | 931/1261 [05:04<01:50,  3.00it/s][A
     74%|███████▍  | 932/1261 [05:05<01:58,  2.78it/s][A
     74%|███████▍  | 933/1261 [05:05<01:53,  2.89it/s][A
     74%|███████▍  | 934/1261 [05:05<01:50,  2.96it/s][A
     74%|███████▍  | 935/1261 [05:06<01:50,  2.94it/s][A
     74%|███████▍  | 936/1261 [05:06<01:51,  2.92it/s][A
     74%|███████▍  | 937/1261 [05:06<02:03,  2.63it/s][A
     74%|███████▍  | 938/1261 [05:07<02:01,  2.67it/s][A
     74%|███████▍  | 939/1261 [05:07<01:54,  2.81it/s][A
     75%|███████▍  | 940/1261 [05:07<01:56,  2.75it/s][A
     75%|███████▍  | 941/1261 [05:08<01:54,  2.79it/s][A
     75%|███████▍  | 942/1261 [05:08<01:49,  2.91it/s][A
     75%|███████▍  | 943/1261 [05:08<01:46,  2.98it/s][A
     75%|███████▍  | 944/1261 [05:09<01:45,  3.00it/s][A
     75%|███████▍  | 945/1261 [05:09<01:46,  2.96it/s][A
     75%|███████▌  | 946/1261 [05:10<01:51,  2.83it/s][A
     75%|███████▌  | 947/1261 [05:10<01:49,  2.88it/s][A
     75%|███████▌  | 948/1261 [05:10<01:42,  3.04it/s][A
     75%|███████▌  | 949/1261 [05:10<01:43,  3.00it/s][A
     75%|███████▌  | 950/1261 [05:11<01:45,  2.94it/s][A
     75%|███████▌  | 951/1261 [05:11<01:41,  3.06it/s][A
     75%|███████▌  | 952/1261 [05:11<01:38,  3.14it/s][A
     76%|███████▌  | 953/1261 [05:12<01:39,  3.09it/s][A
     76%|███████▌  | 954/1261 [05:12<01:40,  3.07it/s][A
     76%|███████▌  | 955/1261 [05:12<01:38,  3.11it/s][A
     76%|███████▌  | 956/1261 [05:13<01:44,  2.91it/s][A
     76%|███████▌  | 957/1261 [05:13<01:43,  2.94it/s][A
     76%|███████▌  | 958/1261 [05:14<01:45,  2.87it/s][A
     76%|███████▌  | 959/1261 [05:14<01:48,  2.78it/s][A
     76%|███████▌  | 960/1261 [05:14<01:45,  2.86it/s][A
     76%|███████▌  | 961/1261 [05:15<01:47,  2.80it/s][A
     76%|███████▋  | 962/1261 [05:15<01:46,  2.81it/s][A
     76%|███████▋  | 963/1261 [05:15<01:42,  2.90it/s][A
     76%|███████▋  | 964/1261 [05:16<01:42,  2.91it/s][A
     77%|███████▋  | 965/1261 [05:16<01:39,  2.98it/s][A
     77%|███████▋  | 966/1261 [05:16<01:37,  3.03it/s][A
     77%|███████▋  | 967/1261 [05:17<01:43,  2.84it/s][A
     77%|███████▋  | 968/1261 [05:17<01:43,  2.82it/s][A
     77%|███████▋  | 969/1261 [05:17<01:41,  2.86it/s][A
     77%|███████▋  | 970/1261 [05:18<01:41,  2.87it/s][A
     77%|███████▋  | 971/1261 [05:18<01:50,  2.63it/s][A
     77%|███████▋  | 972/1261 [05:19<01:52,  2.56it/s][A
     77%|███████▋  | 973/1261 [05:19<01:43,  2.77it/s][A
     77%|███████▋  | 974/1261 [05:19<01:40,  2.86it/s][A
     77%|███████▋  | 975/1261 [05:20<01:46,  2.70it/s][A
     77%|███████▋  | 976/1261 [05:20<01:45,  2.71it/s][A
     77%|███████▋  | 977/1261 [05:20<01:41,  2.80it/s][A
     78%|███████▊  | 978/1261 [05:21<01:38,  2.89it/s][A
     78%|███████▊  | 979/1261 [05:21<01:44,  2.71it/s][A
     78%|███████▊  | 980/1261 [05:21<01:42,  2.74it/s][A
     78%|███████▊  | 981/1261 [05:22<01:44,  2.68it/s][A
     78%|███████▊  | 982/1261 [05:22<01:42,  2.73it/s][A
     78%|███████▊  | 983/1261 [05:22<01:37,  2.85it/s][A
     78%|███████▊  | 984/1261 [05:23<01:39,  2.79it/s][A
     78%|███████▊  | 985/1261 [05:23<01:38,  2.81it/s][A
     78%|███████▊  | 986/1261 [05:24<01:36,  2.84it/s][A
     78%|███████▊  | 987/1261 [05:24<01:39,  2.75it/s][A
     78%|███████▊  | 988/1261 [05:24<01:37,  2.80it/s][A
     78%|███████▊  | 989/1261 [05:25<01:35,  2.86it/s][A
     79%|███████▊  | 990/1261 [05:25<01:35,  2.85it/s][A
     79%|███████▊  | 991/1261 [05:25<01:32,  2.90it/s][A
     79%|███████▊  | 992/1261 [05:26<01:30,  2.97it/s][A
     79%|███████▊  | 993/1261 [05:26<01:36,  2.79it/s][A
     79%|███████▉  | 994/1261 [05:26<01:36,  2.77it/s][A
     79%|███████▉  | 995/1261 [05:27<01:38,  2.70it/s][A
     79%|███████▉  | 996/1261 [05:27<01:36,  2.76it/s][A
     79%|███████▉  | 997/1261 [05:27<01:34,  2.79it/s][A
     79%|███████▉  | 998/1261 [05:28<01:37,  2.70it/s][A
     79%|███████▉  | 999/1261 [05:28<01:36,  2.71it/s][A
     79%|███████▉  | 1000/1261 [05:29<01:38,  2.66it/s][A
     79%|███████▉  | 1001/1261 [05:29<01:41,  2.57it/s][A
     79%|███████▉  | 1002/1261 [05:29<01:41,  2.56it/s][A
     80%|███████▉  | 1003/1261 [05:30<01:37,  2.65it/s][A
     80%|███████▉  | 1004/1261 [05:30<01:34,  2.72it/s][A
     80%|███████▉  | 1005/1261 [05:30<01:27,  2.92it/s][A
     80%|███████▉  | 1006/1261 [05:31<01:23,  3.06it/s][A
     80%|███████▉  | 1007/1261 [05:31<01:26,  2.95it/s][A
     80%|███████▉  | 1008/1261 [05:31<01:30,  2.81it/s][A
     80%|████████  | 1009/1261 [05:32<01:32,  2.74it/s][A
     80%|████████  | 1010/1261 [05:32<01:31,  2.75it/s][A
     80%|████████  | 1011/1261 [05:33<01:32,  2.71it/s][A
     80%|████████  | 1012/1261 [05:33<01:29,  2.79it/s][A
     80%|████████  | 1013/1261 [05:33<01:34,  2.64it/s][A
     80%|████████  | 1014/1261 [05:34<01:28,  2.79it/s][A
     80%|████████  | 1015/1261 [05:34<01:25,  2.88it/s][A
     81%|████████  | 1016/1261 [05:34<01:23,  2.92it/s][A
     81%|████████  | 1017/1261 [05:35<01:25,  2.86it/s][A
     81%|████████  | 1018/1261 [05:35<01:25,  2.85it/s][A
     81%|████████  | 1019/1261 [05:35<01:26,  2.80it/s][A
     81%|████████  | 1020/1261 [05:36<01:25,  2.82it/s][A
     81%|████████  | 1021/1261 [05:36<01:24,  2.83it/s][A
     81%|████████  | 1022/1261 [05:36<01:26,  2.77it/s][A
     81%|████████  | 1023/1261 [05:37<01:25,  2.79it/s][A
     81%|████████  | 1024/1261 [05:37<01:26,  2.75it/s][A
     81%|████████▏ | 1025/1261 [05:38<01:28,  2.67it/s][A
     81%|████████▏ | 1026/1261 [05:38<01:23,  2.80it/s][A
     81%|████████▏ | 1027/1261 [05:38<01:17,  3.03it/s][A
     82%|████████▏ | 1028/1261 [05:38<01:14,  3.11it/s][A
     82%|████████▏ | 1029/1261 [05:39<01:16,  3.03it/s][A
     82%|████████▏ | 1030/1261 [05:39<01:15,  3.07it/s][A
     82%|████████▏ | 1031/1261 [05:40<01:23,  2.74it/s][A
     82%|████████▏ | 1032/1261 [05:40<01:23,  2.74it/s][A
     82%|████████▏ | 1033/1261 [05:40<01:22,  2.75it/s][A
     82%|████████▏ | 1034/1261 [05:41<01:21,  2.78it/s][A
     82%|████████▏ | 1035/1261 [05:41<01:19,  2.86it/s][A
     82%|████████▏ | 1036/1261 [05:41<01:15,  2.98it/s][A
     82%|████████▏ | 1037/1261 [05:42<01:21,  2.73it/s][A
     82%|████████▏ | 1038/1261 [05:42<01:25,  2.62it/s][A
     82%|████████▏ | 1039/1261 [05:43<01:23,  2.67it/s][A
     82%|████████▏ | 1040/1261 [05:43<01:24,  2.63it/s][A
     83%|████████▎ | 1041/1261 [05:43<01:22,  2.67it/s][A
     83%|████████▎ | 1042/1261 [05:44<01:23,  2.62it/s][A
     83%|████████▎ | 1043/1261 [05:44<01:24,  2.59it/s][A
     83%|████████▎ | 1044/1261 [05:44<01:27,  2.49it/s][A
     83%|████████▎ | 1045/1261 [05:45<01:21,  2.64it/s][A
     83%|████████▎ | 1046/1261 [05:45<01:23,  2.57it/s][A
     83%|████████▎ | 1047/1261 [05:46<01:21,  2.61it/s][A
     83%|████████▎ | 1048/1261 [05:46<01:25,  2.50it/s][A
     83%|████████▎ | 1049/1261 [05:46<01:23,  2.55it/s][A
     83%|████████▎ | 1050/1261 [05:47<01:24,  2.49it/s][A
     83%|████████▎ | 1051/1261 [05:47<01:18,  2.66it/s][A
     83%|████████▎ | 1052/1261 [05:47<01:12,  2.86it/s][A
     84%|████████▎ | 1053/1261 [05:48<01:13,  2.84it/s][A
     84%|████████▎ | 1054/1261 [05:48<01:08,  3.00it/s][A
     84%|████████▎ | 1055/1261 [05:48<01:13,  2.82it/s][A
     84%|████████▎ | 1056/1261 [05:49<01:11,  2.86it/s][A
     84%|████████▍ | 1057/1261 [05:49<01:08,  2.97it/s][A
     84%|████████▍ | 1058/1261 [05:50<01:11,  2.85it/s][A
     84%|████████▍ | 1059/1261 [05:50<01:14,  2.71it/s][A
     84%|████████▍ | 1060/1261 [05:50<01:11,  2.80it/s][A
     84%|████████▍ | 1061/1261 [05:51<01:06,  3.02it/s][A
     84%|████████▍ | 1062/1261 [05:51<01:01,  3.25it/s][A
     84%|████████▍ | 1063/1261 [05:51<00:57,  3.46it/s][A
     84%|████████▍ | 1064/1261 [05:51<00:58,  3.36it/s][A
     84%|████████▍ | 1065/1261 [05:52<00:56,  3.46it/s][A
     85%|████████▍ | 1066/1261 [05:52<00:58,  3.32it/s][A
     85%|████████▍ | 1067/1261 [05:52<01:02,  3.12it/s][A
     85%|████████▍ | 1068/1261 [05:53<01:02,  3.08it/s][A
     85%|████████▍ | 1069/1261 [05:53<01:01,  3.10it/s][A
     85%|████████▍ | 1070/1261 [05:53<01:02,  3.07it/s][A
     85%|████████▍ | 1071/1261 [05:54<01:03,  3.00it/s][A
     85%|████████▌ | 1072/1261 [05:54<01:05,  2.90it/s][A
     85%|████████▌ | 1073/1261 [05:54<01:02,  3.03it/s][A
     85%|████████▌ | 1074/1261 [05:55<01:00,  3.10it/s][A
     85%|████████▌ | 1075/1261 [05:55<01:02,  2.98it/s][A
     85%|████████▌ | 1076/1261 [05:55<01:00,  3.06it/s][A
     85%|████████▌ | 1077/1261 [05:56<01:02,  2.94it/s][A
     85%|████████▌ | 1078/1261 [05:56<01:03,  2.89it/s][A
     86%|████████▌ | 1079/1261 [05:56<01:03,  2.86it/s][A
     86%|████████▌ | 1080/1261 [05:57<01:06,  2.71it/s][A
     86%|████████▌ | 1081/1261 [05:57<01:06,  2.70it/s][A
     86%|████████▌ | 1082/1261 [05:58<01:07,  2.67it/s][A
     86%|████████▌ | 1083/1261 [05:58<01:06,  2.70it/s][A
     86%|████████▌ | 1084/1261 [05:58<01:04,  2.76it/s][A
     86%|████████▌ | 1085/1261 [05:59<01:06,  2.66it/s][A
     86%|████████▌ | 1086/1261 [05:59<01:03,  2.77it/s][A
     86%|████████▌ | 1087/1261 [05:59<01:04,  2.70it/s][A
     86%|████████▋ | 1088/1261 [06:00<01:05,  2.65it/s][A
     86%|████████▋ | 1089/1261 [06:00<01:04,  2.68it/s][A
     86%|████████▋ | 1090/1261 [06:01<01:05,  2.61it/s][A
     87%|████████▋ | 1091/1261 [06:01<00:59,  2.84it/s][A
     87%|████████▋ | 1092/1261 [06:01<01:00,  2.81it/s][A
     87%|████████▋ | 1093/1261 [06:02<01:02,  2.70it/s][A
     87%|████████▋ | 1094/1261 [06:02<01:03,  2.64it/s][A
     87%|████████▋ | 1095/1261 [06:02<00:59,  2.79it/s][A
     87%|████████▋ | 1096/1261 [06:03<00:56,  2.93it/s][A
     87%|████████▋ | 1097/1261 [06:03<00:56,  2.90it/s][A
     87%|████████▋ | 1098/1261 [06:03<00:57,  2.85it/s][A
     87%|████████▋ | 1099/1261 [06:04<00:57,  2.84it/s][A
     87%|████████▋ | 1100/1261 [06:04<00:55,  2.89it/s][A
     87%|████████▋ | 1101/1261 [06:04<00:55,  2.90it/s][A
     87%|████████▋ | 1102/1261 [06:05<00:55,  2.86it/s][A
     87%|████████▋ | 1103/1261 [06:05<00:54,  2.87it/s][A
     88%|████████▊ | 1104/1261 [06:05<00:53,  2.92it/s][A
     88%|████████▊ | 1105/1261 [06:06<00:53,  2.92it/s][A
     88%|████████▊ | 1106/1261 [06:06<00:52,  2.96it/s][A
     88%|████████▊ | 1107/1261 [06:06<00:51,  2.97it/s][A
     88%|████████▊ | 1108/1261 [06:07<00:55,  2.78it/s][A
     88%|████████▊ | 1109/1261 [06:07<00:54,  2.81it/s][A
     88%|████████▊ | 1110/1261 [06:08<00:54,  2.76it/s][A
     88%|████████▊ | 1111/1261 [06:08<00:51,  2.91it/s][A
     88%|████████▊ | 1112/1261 [06:08<00:50,  2.93it/s][A
     88%|████████▊ | 1113/1261 [06:08<00:48,  3.08it/s][A
     88%|████████▊ | 1114/1261 [06:09<00:53,  2.76it/s][A
     88%|████████▊ | 1115/1261 [06:09<00:54,  2.69it/s][A
     89%|████████▊ | 1116/1261 [06:10<00:52,  2.77it/s][A
     89%|████████▊ | 1117/1261 [06:10<00:50,  2.84it/s][A
     89%|████████▊ | 1118/1261 [06:10<00:50,  2.82it/s][A
     89%|████████▊ | 1119/1261 [06:11<00:50,  2.81it/s][A
     89%|████████▉ | 1120/1261 [06:11<00:51,  2.72it/s][A
     89%|████████▉ | 1121/1261 [06:11<00:50,  2.76it/s][A
     89%|████████▉ | 1122/1261 [06:12<00:52,  2.64it/s][A
     89%|████████▉ | 1123/1261 [06:12<00:55,  2.49it/s][A
     89%|████████▉ | 1124/1261 [06:13<00:52,  2.63it/s][A
     89%|████████▉ | 1125/1261 [06:13<00:52,  2.59it/s][A
     89%|████████▉ | 1126/1261 [06:13<00:49,  2.72it/s][A
     89%|████████▉ | 1127/1261 [06:14<00:47,  2.84it/s][A
     89%|████████▉ | 1128/1261 [06:14<00:47,  2.78it/s][A
     90%|████████▉ | 1129/1261 [06:14<00:44,  2.98it/s][A
     90%|████████▉ | 1130/1261 [06:15<00:42,  3.07it/s][A
     90%|████████▉ | 1131/1261 [06:15<00:44,  2.92it/s][A
     90%|████████▉ | 1132/1261 [06:15<00:42,  3.01it/s][A
     90%|████████▉ | 1133/1261 [06:16<00:42,  3.02it/s][A
     90%|████████▉ | 1134/1261 [06:16<00:46,  2.75it/s][A
     90%|█████████ | 1135/1261 [06:16<00:45,  2.75it/s][A
     90%|█████████ | 1136/1261 [06:17<00:45,  2.75it/s][A
     90%|█████████ | 1137/1261 [06:17<00:44,  2.79it/s][A
     90%|█████████ | 1138/1261 [06:18<00:46,  2.67it/s][A
     90%|█████████ | 1139/1261 [06:18<00:42,  2.88it/s][A
     90%|█████████ | 1140/1261 [06:18<00:41,  2.93it/s][A
     90%|█████████ | 1141/1261 [06:19<00:39,  3.03it/s][A
     91%|█████████ | 1142/1261 [06:19<00:38,  3.10it/s][A
     91%|█████████ | 1143/1261 [06:19<00:38,  3.08it/s][A
     91%|█████████ | 1144/1261 [06:20<00:40,  2.91it/s][A
     91%|█████████ | 1145/1261 [06:20<00:40,  2.87it/s][A
     91%|█████████ | 1146/1261 [06:20<00:41,  2.74it/s][A
     91%|█████████ | 1147/1261 [06:21<00:41,  2.76it/s][A
     91%|█████████ | 1148/1261 [06:21<00:38,  2.90it/s][A
     91%|█████████ | 1149/1261 [06:21<00:36,  3.05it/s][A
     91%|█████████ | 1150/1261 [06:22<00:36,  3.02it/s][A
     91%|█████████▏| 1151/1261 [06:22<00:34,  3.20it/s][A
     91%|█████████▏| 1152/1261 [06:22<00:35,  3.11it/s][A
     91%|█████████▏| 1153/1261 [06:22<00:34,  3.13it/s][A
     92%|█████████▏| 1154/1261 [06:23<00:35,  3.01it/s][A
     92%|█████████▏| 1155/1261 [06:23<00:36,  2.94it/s][A
     92%|█████████▏| 1156/1261 [06:24<00:35,  2.93it/s][A
     92%|█████████▏| 1157/1261 [06:24<00:31,  3.27it/s][A
     92%|█████████▏| 1158/1261 [06:24<00:30,  3.40it/s][A
     92%|█████████▏| 1159/1261 [06:24<00:29,  3.44it/s][A
     92%|█████████▏| 1160/1261 [06:25<00:30,  3.36it/s][A
     92%|█████████▏| 1161/1261 [06:25<00:29,  3.39it/s][A
     92%|█████████▏| 1162/1261 [06:25<00:30,  3.27it/s][A
     92%|█████████▏| 1163/1261 [06:26<00:29,  3.33it/s][A
     92%|█████████▏| 1164/1261 [06:26<00:31,  3.12it/s][A
     92%|█████████▏| 1165/1261 [06:26<00:30,  3.19it/s][A
     92%|█████████▏| 1166/1261 [06:27<00:32,  2.89it/s][A
     93%|█████████▎| 1167/1261 [06:27<00:32,  2.92it/s][A
     93%|█████████▎| 1168/1261 [06:27<00:32,  2.87it/s][A
     93%|█████████▎| 1169/1261 [06:28<00:31,  2.94it/s][A
     93%|█████████▎| 1170/1261 [06:28<00:32,  2.81it/s][A
     93%|█████████▎| 1171/1261 [06:28<00:30,  2.93it/s][A
     93%|█████████▎| 1172/1261 [06:29<00:29,  2.98it/s][A
     93%|█████████▎| 1173/1261 [06:29<00:31,  2.82it/s][A
     93%|█████████▎| 1174/1261 [06:29<00:30,  2.81it/s][A
     93%|█████████▎| 1175/1261 [06:30<00:31,  2.75it/s][A
     93%|█████████▎| 1176/1261 [06:30<00:32,  2.64it/s][A
     93%|█████████▎| 1177/1261 [06:31<00:31,  2.65it/s][A
     93%|█████████▎| 1178/1261 [06:31<00:31,  2.61it/s][A
     93%|█████████▎| 1179/1261 [06:31<00:31,  2.59it/s][A
     94%|█████████▎| 1180/1261 [06:32<00:30,  2.63it/s][A
     94%|█████████▎| 1181/1261 [06:32<00:30,  2.61it/s][A
     94%|█████████▎| 1182/1261 [06:32<00:28,  2.77it/s][A
     94%|█████████▍| 1183/1261 [06:33<00:27,  2.84it/s][A
     94%|█████████▍| 1184/1261 [06:33<00:28,  2.71it/s][A
     94%|█████████▍| 1185/1261 [06:34<00:28,  2.71it/s][A
     94%|█████████▍| 1186/1261 [06:34<00:28,  2.64it/s][A
     94%|█████████▍| 1187/1261 [06:34<00:27,  2.74it/s][A
     94%|█████████▍| 1188/1261 [06:35<00:26,  2.74it/s][A
     94%|█████████▍| 1189/1261 [06:35<00:25,  2.77it/s][A
     94%|█████████▍| 1190/1261 [06:35<00:25,  2.74it/s][A
     94%|█████████▍| 1191/1261 [06:36<00:25,  2.73it/s][A
     95%|█████████▍| 1192/1261 [06:36<00:25,  2.66it/s][A
     95%|█████████▍| 1193/1261 [06:36<00:23,  2.88it/s][A
     95%|█████████▍| 1194/1261 [06:37<00:23,  2.83it/s][A
     95%|█████████▍| 1195/1261 [06:37<00:24,  2.73it/s][A
     95%|█████████▍| 1196/1261 [06:38<00:23,  2.76it/s][A
     95%|█████████▍| 1197/1261 [06:38<00:23,  2.73it/s][A
     95%|█████████▌| 1198/1261 [06:38<00:22,  2.79it/s][A
     95%|█████████▌| 1199/1261 [06:39<00:21,  2.86it/s][A
     95%|█████████▌| 1200/1261 [06:39<00:20,  2.91it/s][A
     95%|█████████▌| 1201/1261 [06:39<00:19,  3.11it/s][A
     95%|█████████▌| 1202/1261 [06:40<00:19,  3.01it/s][A
     95%|█████████▌| 1203/1261 [06:40<00:19,  2.96it/s][A
     95%|█████████▌| 1204/1261 [06:40<00:19,  2.95it/s][A
     96%|█████████▌| 1205/1261 [06:41<00:19,  2.87it/s][A
     96%|█████████▌| 1206/1261 [06:41<00:20,  2.66it/s][A
     96%|█████████▌| 1207/1261 [06:41<00:20,  2.67it/s][A
     96%|█████████▌| 1208/1261 [06:42<00:19,  2.76it/s][A
     96%|█████████▌| 1209/1261 [06:42<00:19,  2.69it/s][A
     96%|█████████▌| 1210/1261 [06:43<00:19,  2.59it/s][A
     96%|█████████▌| 1211/1261 [06:43<00:20,  2.48it/s][A
     96%|█████████▌| 1212/1261 [06:43<00:18,  2.60it/s][A
     96%|█████████▌| 1213/1261 [06:44<00:18,  2.66it/s][A
     96%|█████████▋| 1214/1261 [06:44<00:17,  2.61it/s][A
     96%|█████████▋| 1215/1261 [06:44<00:16,  2.79it/s][A
     96%|█████████▋| 1216/1261 [06:45<00:16,  2.80it/s][A
     97%|█████████▋| 1217/1261 [06:45<00:16,  2.67it/s][A
     97%|█████████▋| 1218/1261 [06:46<00:15,  2.71it/s][A
     97%|█████████▋| 1219/1261 [06:46<00:15,  2.69it/s][A
     97%|█████████▋| 1220/1261 [06:46<00:14,  2.80it/s][A
     97%|█████████▋| 1221/1261 [06:47<00:13,  2.91it/s][A
     97%|█████████▋| 1222/1261 [06:47<00:12,  3.11it/s][A
     97%|█████████▋| 1223/1261 [06:47<00:12,  3.01it/s][A
     97%|█████████▋| 1224/1261 [06:48<00:12,  3.01it/s][A
     97%|█████████▋| 1225/1261 [06:48<00:12,  2.92it/s][A
     97%|█████████▋| 1226/1261 [06:48<00:12,  2.87it/s][A
     97%|█████████▋| 1227/1261 [06:49<00:12,  2.82it/s][A
     97%|█████████▋| 1228/1261 [06:49<00:11,  2.84it/s][A
     97%|█████████▋| 1229/1261 [06:49<00:10,  2.99it/s][A
     98%|█████████▊| 1230/1261 [06:50<00:10,  2.94it/s][A
     98%|█████████▊| 1231/1261 [06:50<00:09,  3.06it/s][A
     98%|█████████▊| 1232/1261 [06:50<00:10,  2.85it/s][A
     98%|█████████▊| 1233/1261 [06:51<00:09,  2.98it/s][A
     98%|█████████▊| 1234/1261 [06:51<00:08,  3.07it/s][A
     98%|█████████▊| 1235/1261 [06:51<00:08,  2.91it/s][A
     98%|█████████▊| 1236/1261 [06:52<00:08,  2.90it/s][A
     98%|█████████▊| 1237/1261 [06:52<00:08,  2.88it/s][A
     98%|█████████▊| 1238/1261 [06:52<00:07,  3.01it/s][A
     98%|█████████▊| 1239/1261 [06:53<00:07,  2.88it/s][A
     98%|█████████▊| 1240/1261 [06:53<00:07,  2.85it/s][A
     98%|█████████▊| 1241/1261 [06:54<00:07,  2.60it/s][A
     98%|█████████▊| 1242/1261 [06:54<00:07,  2.63it/s][A
     99%|█████████▊| 1243/1261 [06:54<00:06,  2.61it/s][A
     99%|█████████▊| 1244/1261 [06:55<00:06,  2.61it/s][A
     99%|█████████▊| 1245/1261 [06:55<00:05,  2.71it/s][A
     99%|█████████▉| 1246/1261 [06:55<00:05,  2.75it/s][A
     99%|█████████▉| 1247/1261 [06:56<00:05,  2.61it/s][A
     99%|█████████▉| 1248/1261 [06:56<00:04,  2.69it/s][A
     99%|█████████▉| 1249/1261 [06:56<00:04,  2.71it/s][A
     99%|█████████▉| 1250/1261 [06:57<00:03,  2.85it/s][A
     99%|█████████▉| 1251/1261 [06:57<00:03,  3.01it/s][A
     99%|█████████▉| 1252/1261 [06:57<00:03,  2.92it/s][A
     99%|█████████▉| 1253/1261 [06:58<00:02,  2.86it/s][A
     99%|█████████▉| 1254/1261 [06:58<00:02,  2.99it/s][A
    100%|█████████▉| 1255/1261 [06:59<00:02,  2.73it/s][A
    100%|█████████▉| 1256/1261 [06:59<00:01,  2.74it/s][A
    100%|█████████▉| 1257/1261 [06:59<00:01,  2.75it/s][A
    100%|█████████▉| 1258/1261 [07:00<00:01,  2.71it/s][A
    100%|█████████▉| 1259/1261 [07:00<00:00,  2.70it/s][A
    100%|█████████▉| 1260/1261 [07:00<00:00,  2.70it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output1.mp4 
    
    CPU times: user 12min 31s, sys: 1min 1s, total: 13min 32s
    Wall time: 7min 3s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="output1.mp4">
</video>




### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Deciding over the different kinds of thresholding approaches was not easy, intuitive approaches were good for single images, but were sometimes failing on the video.

The pipeline will likely fail when there are large amounts of paint spilled on the road, near the lines.

To make it more robust, a different kind of threshold can be implemented. Also, machine learning approaches can be used for deciding over different kinds of filters on different conditions.

This was a nice project with much fun, but I understand better the importance of behavioral cloning approach now. :)


```python

```


```python

```


```python

```


```python

```


```python

```
