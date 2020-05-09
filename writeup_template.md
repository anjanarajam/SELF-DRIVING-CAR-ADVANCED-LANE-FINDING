## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![chessboard_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/calibration1.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![undistorted_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/undist_image.png)

When I calibrated the camera I got the camera matrix and the distortion coefficients. These two parameters are used to get the undistorted image by using function undistort_image().

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Canny edge detects almost everything in image. How do we take advantage of the fact that lane lines are vertical. We use gradients to detect steep edges. Canny edge takes derivative of x and y to detect edges. In order to take derivative of x and y, Sobel operator is used. When you superimpose sobel operator on a region, you get the product of the matrix. If the sum of the resultant matrix is in x direction, emphasises edges closer to vertical and in y direction, emphasises edges closer to horizontal.

I form a binary image within a threshold so that pixels are selected based on gradient strength. I have applied thrshold to overall magnitude of the gradient, in both x and y direction. Next I considered colour spaces so that it gives more inforamtion like the colour of the lane line which otherwise wouldnt have been possible with grayscale image alone. In the colour space I chose saturation under the HLS colour space. Saturation is a measure of coloufulness. Colours closer to white have low saturation and intense colours have high saturation. out of all the colour spaces, Saturation shows the clearest of the lane lines, irrespective of shadows and different colours of pavement.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 6 through 60 in `thresholding.py`).Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![combined_binary_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/combined_binary_image.png)

```python
def hls_threshold(image, thresh=(0, 255)):
    """This function applies threshold to saturation colour space to detect the lane line"""
        
    # Convert the image to HLS colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Separate the 'S' channel
    S = hls[:,:,2]
    
    #  Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    return binary_output

def magnitude_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    """This function applies threshold to magnitude gradient to detect the lane line"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # Return this mask as your binary_output image
    return binary_output

def combine_binary(image):
    #Get the binary image of hls threshold
    hls_binary = hls_threshold(image, thresh=(90, 255))
    
    #Get the binary image of magnitude threshold
    mag_binary = magnitude_threshold(image, sobel_kernel=3, thresh=(30, 100))
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(mag_binary), mag_binary, hls_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(mag_binary)
    combined_binary[(hls_binary == 1) | (mag_binary == 1)] = 1
    
    return combined_binary 
 ```

When you combine colour and gradient threshold, we can clearly see parts of lane lines detected by gradient threshold and parts detected by colour threshold by stacking the channels and seeing individual components.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image. 
The use of perspective transform is to find the curved lines, by making it look like in a top down approach, like that of maps, in a zoomed manner. This is called birds eyes view. We select four points to perform a linear transformation from one perspective to another.
We get perspective matrix, M, from function `getPerspectiveTransform()`, which is used to warp the perspective of an image.

The code for my perspective transform includes a function called `warpPerspective()`, which appears in lines 6 through 25 in the file `perspective_transform.py`. The `warpPerspective()` function takes as inputs an image (`threshold_image`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.array([[200,720], [1150, 720], [750, 480], [550, 480]], np.float32)
dst = np.array([[300,720], [900, 720], [900, 0], [300, 0]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 1150, 720     | 900, 720      |
| 750, 480      | 900, 0        |
| 550, 480      | 300, 0        |

```python
def transform_perspective(threshold_image):
    # Define the four source points
    src = np.array([[200,720], [1150, 720], [750, 480], [550, 480]], np.float32)
    
    # Define the four destination points
    dst = np.array([[300,720], [900, 720], [900, 0], [300, 0]], np.float32)
    
    # Get the transformation matrix by performing perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Get the inverse transformation matrix 
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Get the image size 
    image_size = (threshold_image.shape[1], threshold_image.shape[0])
    
    # Warp the image 
    warped_image = cv2.warpPerspective(threshold_image, M, image_size)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![combined_binary_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/warped_image.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
A histogram of the lower half of the image was taken(where lane lines are most likely to be vertical) to find the peak, which are nothing but the left and the right lanes. These peaks are considered as the starting point(x-positions) of the lane lines. From the peak, n number of sliding windows are formed upto the top of the frame with a margin of +/- 100 from starting points. I then, extracted the activated pixels(non-zero x and y points), within the window. These x and y activated pixels are fit in a polynomial using np.polyfit() to get the left and right coefficients respectively. With the help of coefficients and the y value of the image, x values of left and right lane are calculated using the second order polynomial equation.

```python
def find_lane_line_pixels(warped_image):
    # Plot a histogram where the binary activations occur across the image
    # Get the image height
    image_height = warped_image.shape[0]
    # Get the pixel value lower half of the image(half of rows and the complete columns 
    # where lane lines are most likely to be vertical
    lower_half = warped_image[image_height // 2:,:]

    # Get the sum across the vertical line or the height of the image or sum of the columns
    histogram = np.sum(lower_half, axis=0)

    # Plot the histogram
    plt.plot(histogram)

    # Create an output image to draw on and visualize the result
    output_image = np.dstack((warped_image, warped_image, warped_image)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    mid_point = np.int(histogram.shape[0]//2)
    # Maximum value of x in the left side of histogram
    left_x_point = np.argmax(histogram[:mid_point])
     # Maximum value of x in the right side of histogram
    right_x_point = np.argmax(histogram[mid_point:]) + mid_point
    
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    # non zero returns non zero positions in row and column. 
    non_zero_tuple = warped_image.nonzero()
    # nonzero[0] is the array of non zero postions in col(y positions)
    non_zero_y = np.array(non_zero_tuple[0])
    # nonzero[0] is the array of non zero postions in row (x positions)
    non_zero_x = np.array(non_zero_tuple[1])

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin from the windows starting point
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows 
    window_height = np.int(image_height // nwindows)
    
    # Current positions to be updated later for each window in nwindows
    current_left_x_pos = left_x_point
    current_right_x_pos = right_x_point

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Iterate through the number of windows
    for window in range(nwindows):
        # Find the boundaries of our current window.         
        # Find y position of the pixels within the window
        y_low = image_height - (window + 1) * window_height
        y_high = image_height - window * window_height
        
        # Find x position of the pixels within the window
        x_left_low = current_left_x_pos - margin
        x_left_high = current_left_x_pos + margin
        x_right_low = current_right_x_pos - margin
        x_right_high = current_right_x_pos + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(output_image, (x_left_low, y_low), (x_left_high, y_high),(0, 0, 255), 3) 
        cv2.rectangle(output_image, (x_right_low, y_low),(x_right_high, y_high),(0, 0, 255), 3) 
        
        # Identify the nonzero pixels in x and y within the window which means the pixels are highly activated which 
        # will be mostly that of a lane     
        left_window_nonzero_indices = ((non_zero_y >= y_low) & (non_zero_y < y_high) & 
        (non_zero_x >= x_left_low) &  (non_zero_x < x_left_high)).nonzero()[0]
        right_window_nonzero_indices = ((non_zero_y >= y_low) & (non_zero_y < y_high) & 
        (non_zero_x >= x_right_low) &  (non_zero_x < x_right_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_indices.append(left_window_nonzero_indices)
        right_lane_indices.append(right_window_nonzero_indices)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_nonzero_indices) > minpix:
            current_left_x_pos = np.int(np.mean(non_zero_x[left_window_nonzero_indices]))
        if len(right_window_nonzero_indices) > minpix:        
            current_right_x_pos = np.int(np.mean(non_zero_x[right_window_nonzero_indices]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right lane line pixel position within the window
    left_laneline_x_pixels = non_zero_x[left_lane_indices]
    left_laneline_y_pixels = non_zero_y[left_lane_indices] 
    right_laneline_x_pixels = non_zero_x[right_lane_indices]
    right_laneline_y_pixels = non_zero_y[right_lane_indices]

    return left_laneline_x_pixels, left_laneline_y_pixels, right_laneline_x_pixels, right_laneline_y_pixels, output_image


def find_lane_line_from_polynomial(warped_image):
    # Find our lane pixels first
    left_laneline_x_pixels, left_laneline_y_pixels, right_laneline_x_pixels, right_laneline_y_pixels, output_image =                                                                                                        find_lane_line_pixels(warped_image)

    # Find the coefficients of the polynomial formed by polyfit of the left and right lane line pixels 
    # to find the left and the right lane lines
    left_laneline_coeff = np.polyfit(left_laneline_y_pixels, left_laneline_x_pixels, 2)
    right_laneline_coeff = np.polyfit(right_laneline_y_pixels, right_laneline_x_pixels, 2)

    # Generate x and y values for plotting from image vertically or column 
    image_y_values = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    
    try:
        # Find the x values of the lane lines from the polynomials
        left_laneline_x_values = left_laneline_coeff[0] * image_y_values ** 2 + left_laneline_coeff[1] * image_y_values +                                                left_laneline_coeff[2]
        right_laneline_x_values = right_laneline_coeff[0] * image_y_values ** 2 + right_laneline_coeff[1] * image_y_values +                                             right_laneline_coeff[2]
    except TypeError:
        # Avoids an error if left and right coefficients are still none or incorrect
        print('The function failed to fit a line!')
        left_laneline_x_values = 1 * image_y_values ** 2 + 1 * image_y_values
        right_laneline_x_values = 1 * image_y_values ** 2 + 1 * image_y_values

    ## Visualization ##
    # Colors in the left and right lane regions
    output_image[left_laneline_y_pixels, left_laneline_x_pixels] = [255, 0, 0]
    output_image[right_laneline_y_pixels, right_laneline_x_pixels] = [255, 0, 0]

    # Plots the left and right x and y values of the lane lines
    plt.plot(left_laneline_x_values, image_y_values, color='yellow')
    plt.plot(right_laneline_x_values, image_y_values, color='yellow')

    return output_image, (left_laneline_coeff, right_laneline_coeff)
```
![sliding_window_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/sliding_window.png)
![lane_line_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/lane_line.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The radius of curvature is calculated using the equation, which is given in the following link <https://www.intmath.com/applications-differentiation/8-radius-curvature.php>. 

The y values increase from top to bottom of the image. Hence if the curvature has to be calculated the y values at the bottom of the image is considered, which will be ymax. In the real world space, the real dimension of the lane is taken as inputs. According to the U.S. regulations, the minimum lane width should be 12 feet or 3.7 meters and lets take the length of the lane as 30 meters. Since our camera image is in pixels, I convert the pixels in meters as follows:
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

The coefficients are derived using np.polyfit taking image y value and the x pixel values. Now that we have the coeffiecients, ymax value, the left and the right curvature is calculated using the formula.

I did this in lines 3 through 30 in my code in `radius_of_curvature.py`

```python
def get_radius_of_curvature(x_pixels):
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ymeters_per_pixel = 30/720 
    # meters per pixel in x dimension
    xmeters_per_pixel = 3.7/700 
    
    # Get x, y values from image
    y_image_values = np.linspace(0, 719, num=720)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image since 
    # we need curvature closest to the vehicle
    y_max = np.max(y_image_values)
    
    # Get the left and right pixels
    left_x_pixel = x_pixels[0]
    right_x_pixel = x_pixels[1]
    
    # Get the left and right coefficients
    left_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, left_x_pixel * xmeters_per_pixel, 2)
    right_x_coeff = np.polyfit(y_image_values * ymeters_per_pixel, right_x_pixel * xmeters_per_pixel, 2)
       
    # Calculate radius of curvature 
    left_curvature = ((1 + (2* left_x_coeff[0] * y_max * ymeters_per_pixel + left_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                    left_x_coeff[0])
    right_curvature = ((1 + (2 * right_x_coeff[0] * y_max * ymeters_per_pixel + right_x_coeff[1]) ** 2) ** 1.5) / np.absolute(2 *                                 right_x_coeff[0])
    
    return (left_curvature, right_curvature)
 ```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step under function `original_lane_lines()` in `main.py` in the function .  Here is an example of my result on a test image:
![original_lane_line_image](https://github.com/anjanarajam/SELF-DRIVING-CAR-ADVANCED-LANE-FINDING/tree/master/output_images/original_lane_line_image.png)

=--

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
