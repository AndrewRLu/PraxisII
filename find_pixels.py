import cv2

# Load the image
image = cv2.imread(r"C:\Users\andre\Desktop\praxis 2\varroa_test.png")

# Define a function to get pixel location on mouse click
def get_pixel_location(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        print(f"Pixel location: ({x}, {y})")

# Display the image and set up a mouse callback
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', get_pixel_location)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()