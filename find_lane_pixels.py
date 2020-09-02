import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_lane_line_pixels(warped_image):
    ### Plot a histogram where the binary activations occur across the image
    
    # Get the image height
    image_height = warped_image.shape[0]
    # Get the pixel value of the lower half of the image(half of rows and the complete columns 
    # where lane lines are most likely to be vertical
    lower_half = warped_image[image_height // 2:,:]

    # Get the sum across the vertical line or the height of the image or sum of the columns
    histogram = np.sum(lower_half, axis=0)

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

    ### Find x and y non-zero pixels for every window, append it in the list, concatenate and then finally find the non zero x and y 
    ### pixel values of the entire left and right line from the concatenation
    
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
        
        # Identify the nonzero pixels in x and y from each window which means the pixels are highly activated which 
        # will be mostly that of a lane     
        left_window_nonzero_indices = ((non_zero_y >= y_low) & (non_zero_y < y_high) & 
        (non_zero_x >= x_left_low) &  (non_zero_x < x_left_high)).nonzero()[0]
        right_window_nonzero_indices = ((non_zero_y >= y_low) & (non_zero_y < y_high) & 
        (non_zero_x >= x_right_low) &  (non_zero_x < x_right_high)).nonzero()[0]
        
        # Append nonzero pixels in x and y to the lists
        left_lane_indices.append(left_window_nonzero_indices)
        right_lane_indices.append(right_window_nonzero_indices)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_nonzero_indices) > minpix:
            current_left_x_pos = np.int(np.mean(non_zero_x[left_window_nonzero_indices]))
        if len(right_window_nonzero_indices) > minpix:        
            current_right_x_pos = np.int(np.mean(non_zero_x[right_window_nonzero_indices]))

    # Concatenate the arrays of indices. Concatenation helps in combining all
    # the arrays so that we get the aggregate nonzero pixels in x and y for the entire left and right line
    try:
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right lane line pixel position separately from the entire left and the right lines
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

    return output_image, (left_laneline_coeff, right_laneline_coeff), (left_laneline_x_values, right_laneline_x_values)

def find_next_lane_line_from_prev_poly(prev_laneline_coeff, warped_image):
    # width of the margin around the previous polynomial to search
    margin = 100
    
    # Get the left and right lane coefficients from the previous line
    prev_left_laneline_coeff = prev_laneline_coeff[0]
    prev_right_laneline_coeff = prev_laneline_coeff[1]
    
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    # non zero returns non zero positions in row and column. 
    non_zero_tuple = warped_image.nonzero()
    # nonzero[0] is the non zero postions in col(y positions)
    non_zero_y = np.array(non_zero_tuple[0])
    # nonzero[0] is the non zero postions in row (x positions)
    non_zero_x = np.array(non_zero_tuple[1])
    
    # Find the x values from the polynomials based the activated y pixels of the image taking the
    # previous line coefficients. This is because, the first time activated pixels were not known.
    # We had to form window by window to find the pixels and the coefficients. Now we can straight away use coefficients of previous line
    # and non zero y value of the image to find the new x and y value pixels of the new line
    left_laneline_nonzero_x = prev_left_laneline_coeff[0] * non_zero_y ** 2 + prev_left_laneline_coeff[1] * non_zero_y +                                                 prev_left_laneline_coeff[2]
    right_laneline_nonzero_x = prev_right_laneline_coeff[0] * non_zero_y ** 2 + prev_right_laneline_coeff[1] * non_zero_y +                                               prev_right_laneline_coeff[2]
    
    # Find the pixels of the left and right lane pixels within the margin with the x values
    left_lane_indices = ((non_zero_x > (left_laneline_nonzero_x - margin)) & (non_zero_x < (left_laneline_nonzero_x + margin)))
    right_lane_indices = ((non_zero_x > (right_laneline_nonzero_x - margin)) & (non_zero_x < (right_laneline_nonzero_x + margin)))
    
    # Extract new left and right line pixel positions
    left_laneline_x_pixels_new = non_zero_x[left_lane_indices]
    left_laneline_y_pixels_new = non_zero_y[left_lane_indices] 
    right_laneline_x_pixels_new = non_zero_x[right_lane_indices]
    right_laneline_y_pixels_new = non_zero_y[right_lane_indices]
    
    # Now find the new coefficients based on the new left and right lane pixels found      
    left_laneline_coeff_new = np.polyfit(left_laneline_y_pixels_new, left_laneline_x_pixels_new, 2)
    right_laneline_coeff_new = np.polyfit(right_laneline_y_pixels_new, right_laneline_x_pixels_new, 2)
    
    # Generate x and y values for plotting
    image_y_values = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    
    # Calculate left and right values of x using image_y_values, left and right coefficients in the polynomial equation
    left_laneline_x_values_new = left_laneline_coeff_new[0] * image_y_values ** 2 + left_laneline_coeff_new[1] * image_y_values +                                            left_laneline_coeff_new[2]
    right_laneline_x_values_new = right_laneline_coeff_new[0]* image_y_values ** 2 + right_laneline_coeff_new[1] * image_y_values +                                           right_laneline_coeff_new[2]

    # Create an image to draw on and an image to show the selection window
    output_image = np.dstack((warped_image, warped_image, warped_image)) * 255
    window_image = np.zeros_like(output_image)
    
    # Color the left and right line pixels
    output_image[left_laneline_y_pixels_new, left_laneline_x_pixels_new] = [255, 0, 0]
    output_image[right_laneline_y_pixels_new, right_laneline_x_pixels_new] = [0, 0, 255]
                 
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_laneline_x_values_new - margin, image_y_values]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_laneline_x_values_new + margin, image_y_values])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_laneline_x_values_new - margin, image_y_values]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_laneline_x_values_new + margin, image_y_values])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(output_image, 1, window_image, 0.3, 0)
    
    return result, (left_laneline_coeff_new, right_laneline_coeff_new), (left_laneline_x_values_new, right_laneline_x_values_new)


   
    




