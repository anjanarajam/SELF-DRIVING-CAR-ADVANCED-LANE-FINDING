import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gradient_threshold(image, orient='x', thresh_min=0, thresh_max=255):
    """This function applies threshold to x, y gradient to detect the lane line"""    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        # Take the derivative in x"
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    elif orient == 'y':
        # Take the derivative in y"
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Return this mask as your binary_output image
    return binary_output   

def hls_threshold(image, channel, thresh=(0, 255)):
    """This function applies threshold to saturation colour space to detect the lane line"""
        
    # Convert the image to HLS colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    if channel == 'S':
        # Separate the 'S' channel
        C = hls[:,:,2]
    elif channel == 'L':
        # Separate the 'L' channel
        C = hls[:,:,1]
    elif channel == 'H':
        # Separate the 'H' channel
        C = hls[:,:,0]
    
    #  Apply a threshold to the S channel
    binary_output = np.zeros_like(C)
    binary_output[(C > thresh[0]) & (C <= thresh[1])] = 1
    
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

def direction_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """This function applies threshold to direction gradient to detect the lane line"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    
    # Return this mask as your binary_output image
    return binary_output

def combine_binary(image):
    # Get the binary for the x gradients 
    x_binary = gradient_threshold(image, orient='x', thresh_min=20, thresh_max=200)
    
    # Get the binary image of hls threshold to see yellow light 
    # Saturation is high for bright colours but low for faded colour
    # Lightness is low for bright but high for faded colours
    S_binary = hls_threshold(image, channel='S', thresh=(120, 255))    
    L_binary = hls_threshold(image, channel='L', thresh=(35, 210))    
    H_binary = hls_threshold(image, channel='H', thresh=(15, 255))      
    
    hls_binary_yellow = np.zeros_like(S_binary)    
    hls_binary_yellow[(S_binary == 1) & (L_binary == 1) & (L_binary == 1)] = 1
    
    # Get the binary image of hls threshold to see white light 
    S_binary = hls_threshold(image, channel='S', thresh=(0, 255))    
    L_binary = hls_threshold(image, channel='L', thresh=(200, 255))    
    H_binary = hls_threshold(image, channel='H', thresh=(0, 255)) 
    
    hls_binary_white = np.zeros_like(S_binary)    
    hls_binary_white[(S_binary == 1) & (L_binary == 1) & (L_binary == 1)] = 1
    
    # Combine binary for yellow and white colour
    hls_binary = np.zeros_like(S_binary) 
    hls_binary[(hls_binary_yellow == 1) | (hls_binary_white == 1)] = 1
    
    # Get the binary image of magnitude threshold
    mag_binary = magnitude_threshold(image, sobel_kernel=5, thresh=(80, 200))
    
#     Get the binary image of magnitude threshold
    dir_binary = direction_threshold(image, sobel_kernel=5, thresh=(0, np.pi/2))
    
    combined_gradient_binary = np.zeros_like(mag_binary) 
    combined_gradient_binary[(x_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] == 1
#     combined_gradient_binary[((x_binary == 1) | (mag_binary == 1))] == 1
    
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(combined_gradient_binary), combined_gradient_binary, hls_binary)) * 255
#     color_binary = np.dstack(( np.zeros_like(mag_binary), mag_binary, hls_binary)) * 255
    
#     # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined_gradient_binary)
    combined_binary[(hls_binary == 1) | (combined_gradient_binary == 1)] = 1

#     # Combine the two binary thresholds
#     combined_binary = np.zeros_like(mag_binary)
#     combined_binary[(hls_binary == 1) | (mag_binary == 1)] = 1
    
    # apply the region of interest mask
    mask = np.zeros_like(combined_binary)
    region_of_interest = np.array([[0, image.shape[0]-1], [image.shape[1] / 2, int(0.5 * image.shape[0])], [image.shape[1]-1, image.shape[0]-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest], 1)
    threshold_image = cv2.bitwise_and(combined_binary, mask)
    
    return threshold_image 
    