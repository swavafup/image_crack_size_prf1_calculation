import os
import cv2
import numpy as np

# Path to the folder containing 2D slice images
folder_path = 'C:/Users/swava/Desktop/CRACK-SAM/Dataset2/masksam_gt/'

# Function to calculate the area of the cracks in a single binary slice image
def calculate_crack_area(slice_binary_image):
    # Find contours of the crack regions
    contours, _ = cv2.findContours(slice_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store crack areas
    crack_areas = []
    
    # Calculate the area of the cracks
    total_crack_area = 0
    for contour in contours:
        crack_area = cv2.contourArea(contour)
        total_crack_area += crack_area
        crack_areas.append(crack_area)
    
    return total_crack_area, crack_areas

# Load a single binary image for demonstration
filename = 'CRACK500_20160222_164141_1281_1.png'
slice_binary_image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)

# Calculate the area of the cracks for the single image
total_crack_area, crack_areas = calculate_crack_area(slice_binary_image)

# Print the results
print(f'Total Crack Area: {total_crack_area}')
print(f'Crack Areas: {crack_areas}')
