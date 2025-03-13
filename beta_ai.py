import cv2 as cv
import numpy as np
import math as m

# Load an image
image = cv.imread(r"C:\Users\andre\Desktop\praxis 2\test_red.jpg")

# Convert the image to HSV color space
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define the lower and upper bounds for red
lower_red1 = np.array([0, 50, 50])  # Lower range (0-10)
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 50, 50])  # Upper range (170-180)
upper_red2 = np.array([180, 255, 255])

# Create masks for both ranges
mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)

# Combine the masks
red_mask = cv.bitwise_or(mask1, mask2)

# Calculate the size of the mask (number of non-zero pixels)
mask_size = cv.countNonZero(red_mask)

# Calculate the size of the entire image (total number of pixels)
image_size = image.shape[0] * image.shape[1]  # height * width

# Compute the ratio of mask size to image size
ratio = mask_size / image_size

#number of things
acc_num = 10
size_each = m.pi*((image.shape[0]*0.1/2)**2)
calc_r = mask_size / size_each

# Print the results
print(f"Size of mask: {mask_size} pixels")
print(f"Size of entire image: {image_size} pixels")
print(f"Ratio of mask to image: {ratio:.4f}")
print(f"Calculated num: {calc_r:.4f}, Acutal num: {acc_num}, size each: {size_each:.4f}")

# Display the results
cv.imshow("Red Mask", red_mask)
cv.imshow("Red Objects", cv.bitwise_and(image, image, mask=red_mask))

cv.waitKey(0)
cv.destroyAllWindows()