# **Advanced Lane Finding Project** 
by Hanbyul Yang, Sep 19, 2017

## Overview

This is a project of Self-Driving Car Nanodegree Program of Udacity.

The goals of this project is finding lane lines of given image or videos of driving car. 
Details of goals and steps are following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

For the processing pipelines and codes, Check [`P4.ipynb`](./P4.ipynb).
I wrote this in the order given [rubrics](https://review.udacity.com/#!/rubrics/571/view).

[//]: # (Image References)

[camera_calibration]: ./output_images/01_camera_calibration.png "calibration test"
[test_image_n_undistorted]: ./output_images/02_undistoted_test_image.png "test image and undistorted"
[pipeline_out]: ./output_images/03_pipeline_out.png "Binary Example"
[warped]: ./output_images/04_perspective_transform.png "Warp Example"
[fitted]: ./output_images/05_fit_lines.png "Fitted lines"
[output]: ./output_images/06_output.png "Output example"
[output_video]: ./output.mp4 "Video"


## Writeup / README
This file `writeup.md` is for writeup. `README.md` describes contents (files and folders) briefly. 


## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd cell of the jupyter notebook located in [`./P4.ipynb`](./P4.ipynb)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][camera_calibration]

## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

I applied the distortion correction to one of the test images located in `./examples/test3.jpg`. belows are original test image and camera calibrated image.
![alt text][test_image_n_undistorted]

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I transformed image from RGB to HLS representation for robust detection of lanes.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at 5th cell of [`P4.ipynb`](./P4.ipynb)). S channel is used for color threshold and L channel for x gradient threshold.
 Here's an example of my output for this step. 
![alt text][pipeline_out]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in 6th cell of [`P4.ipynb`](./P4.ipynb).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose manually the source and destination points which is given in example code as following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
Red line shows that perspective transform was done well as expected.

![alt text][warped]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used "Peaks in histogram" method as line finding method. I first take a histogram along all the columns in the lower half of the image. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. Used that point as a starting point for where to search for the lines. I divided height of images with 9 part. Then, I fit 2nd order polynomials with found points. The code for finding lines is at 9th cell of [`P4.ipynb`](./P4.ipynb)

![alt text][fitted]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of curvature of the lane is performed by function located in 12th cell and used it in the 71st line of 13th cell of the jupyter notebook. The y-value where we want radius of curvature is chosen by the the maximum y-value, corresponding to the bottom of the image. Two radius of curvature for left and right lane are calculated. I determined radius of curvature by averaging those two radius. Converting pixel to meter is performed by multiplying given constant ratio. 

The position of vehicle with respect to center is in the process() function located in 62nd line of 14th cell of [jupyter notebook](./P4.ipynb). The y-value where we want radius of curvature is chosen by the the maximum y-value. 


### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 14th cell of [jupyter notebook](./P4.ipynb) in the function `process()`. Lane is colored with green, left lane with red and right lane with blue.

![alt text][output]


## Pipeline (video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The 16th cell of [jupyter notebook](./P4.ipynb) shows the process creating video.
Here's a [link to my video result](./output.mp4)


## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In project video, One problem I met with was when car is on the road where color of road is changing. It occurs from 20 to 25 seconds and 39 ~ 42 sec. The other problem I met was the shadow of trees on the road. I applied several sanity checks. For example radius cannot be smaller of bigger than certain ratio of previous radius. position of line base which is bottom of the image cannot be moved some amount by frame. If one of sanity checks were failed, I used previous result for that frame. Furthermore, 5 fails in a row, I reset all previous result and find lines from scratch by using peaks in histogram and sliding window method. Also, for more robustness, I applied temporal filtering by averaging 4 previous fitted results and current fitted result. this gives smooth movement of lane finding.
