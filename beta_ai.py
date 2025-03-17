import cv2 as cv
import numpy as np

# Load an image
image = cv.imread(r"C:\Users\andre\Desktop\praxis 2\varroa_test.png")

# Normalize the image to the range [0, 255]
normalized_img = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

# Convert to LAB color space for CLAHE
lab_image = cv.cvtColor(normalized_img, cv.COLOR_BGR2LAB)

# Split LAB channels
l, a, b = cv.split(lab_image)

# Apply CLAHE to the L channel
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_l = clahe.apply(l)

# Merge the CLAHE-enhanced L channel back with the original A and B channels
enhanced_lab = cv.merge((clahe_l, a, b))

# Convert back to RGB color space
enhanced_rgb = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

# Define the lower and upper bounds for red in RGB
lower_red = np.array([1, 0, 0])  # Red ≥ 150, Green ≥ 0, Blue ≥ 0
upper_red = np.array([255, 100, 100])  # Red ≤ 255, Green ≤ 100, Blue ≤ 100

# Create a mask using the RGB color space
red_mask = cv.inRange(enhanced_rgb, lower_red, upper_red)

# Find contours in the mask
contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create an empty mask to store the final result
final_mask = np.zeros_like(red_mask)

# Define the maximum allowed size for objects (30x30 pixels)
max_size = 30
min_size = 10

# Loop through the contours
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv.boundingRect(contour)
    
    # Check if the object is larger than the maximum size
    if w > max_size or h > max_size:
        # Create a mask for the current contour
        contour_mask = np.zeros_like(red_mask)
        cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
        
        # Display the contour mask and the original image
        cv.imshow("yay or nay", cv.bitwise_and(enhanced_rgb, enhanced_rgb, mask=contour_mask))
        
        # Prompt the user for input
        print("Object detected with size:", (w, h))
        key = cv.waitKey(0)  # Wait for user input
        
        # If the user presses 'y', include the object in the final mask
        if key == ord('y'):
            cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)
            print("Object included in the final mask.")
        else:
            print("Object excluded from the final mask.")
        
        # Close the contour mask window
        cv.destroyWindow("yay or nay")
    else:
        # Automatically include small objects in the final mask
        if w > min_size or h > min_size: #get rid of small specs
            cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)

#math
image_size = image.shape[0] * image.shape[1]
mask_size = cv.countNonZero(final_mask)
area_mite = 20*20
calc_r = mask_size / area_mite
acc_num = 103


# Display the final results
cv.imshow("Mites Mask", cv.bitwise_and(enhanced_rgb, enhanced_rgb, mask=final_mask))
cv.imshow("Original", image)
print(f"Calculated num: {calc_r:.4f}, Acutal num: {acc_num}")

cv.waitKey(0)
cv.destroyAllWindows()