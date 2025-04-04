'''
this version uses a reference
'''

import cv2 as cv
import numpy as np
import requests
import time

height = 1080
width = 1920

url = "my url"

def get_image_from_ipcam(url):
    print("Press 'y' to capture, 'q' to quit")
    
    while True:
        # Get current frame
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv.imdecode(img_arr, -1)
        resized_img = cv.resize(img, (1280, 720), interpolation=cv.INTER_AREA)
        
        # Show live preview
        cv.imshow("IP Camera Feed", resized_img)
        
        # Check for key press (non-blocking)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv.destroyAllWindows()
            return None
        elif key == ord('y'):
            # Show the captured frame for confirmation
            print("Press 'y' to SAVE picture, or any other key to TRY AGAIN!...")
            cv.imshow("Confirm Capture - Press 'y' to save, any key to cancel", resized_img)
            confirm_key = cv.waitKey(0) & 0xFF
            
            if confirm_key == ord('y'):
                cv.destroyAllWindows()
                return resized_img
            else:
                # If not confirmed, return to live view
                cv.destroyWindow("Confirm Capture - Press 'y' to save, any key to cancel")

#ref
def reference_photo(url):
    print("1. Take a reference photo:")
    ref = get_image_from_ipcam(url)
    print("Reference photo saved!")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"captured_{timestamp}.png"
    cv.imwrite(image_filename, ref)
    print(f"Image captured and saved as {image_filename}")
    return ref

def get_contours(url): # fix after
    print("2. Take a picture of your sticky board:")
    image = get_image_from_ipcam(url) 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"captured_{timestamp}.png"
    cv.imwrite(image_filename, image)
    print(f"Image captured and saved as {image_filename}")
    return image

def calc_ref_area(image):
    # Normalize the image to the range [0, 255]
    normalized_img = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Convert to HSV color space for better color segmentation
    hsv_image = cv.cvtColor(normalized_img, cv.COLOR_BGR2HSV)

    # Define ranges for red color in HSV
    # Red is tricky because it wraps around 0 in HSV
    lower_red1 = np.array([0, 50, 50])    # Lower range for hue (0-10)
    upper_red1 = np.array([15, 255, 255])   # Upper range for hue (0-10)
    lower_red2 = np.array([160, 50, 50])  # Lower range for hue (160-180)
    upper_red2 = np.array([180, 255, 255])  # Upper range for hue (160-180)

    # Create masks for both red ranges
    mask1 = cv.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        if cv.contourArea(contour) > 20: # tweak this number for reference

            # Create a mask for the current contour
            contour_mask = np.zeros_like(red_mask)
            cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
            
            # Display the contour mask and the original image
            temp = cv.bitwise_and(hsv_image, hsv_image, mask=contour_mask)
            display_image = cv.cvtColor(temp, cv.COLOR_HSV2BGR)
            cv.imshow("good ref?", display_image)
            print("Does this look like a mite?")

            key = cv.waitKey(0)
            if key == ord('y'):
                print("Great, reference mask saved!")
                area = cv.contourArea(contour)
                return area
            
            else:
                print("Let's try again,")

def mite_calculation(image, ref_area):

    total_area = 0
    total_objects = 0

    # Normalize the image to the range [0, 255]
    normalized_img = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Convert to HSV color space for better color segmentation
    hsv_image = cv.cvtColor(normalized_img, cv.COLOR_BGR2HSV)
    # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # # Define ranges for red color in HSV
    # # Red is tricky because it wraps around 0 in HSV
    # lower_red1 = np.array([0, 50, 50])    # Lower range for hue (0-10)
    # upper_red1 = np.array([15, 255, 255])   # Upper range for hue (0-10)
    # lower_red2 = np.array([160, 50, 50])  # Lower range for hue (160-180)
    # upper_red2 = np.array([180, 255, 255])  # Upper range for hue (160-180)

    # lower_red1 = np.array([0, 30, 30])     # Hue: 0-20, lower saturation/value for darker colors
    # upper_red1 = np.array([20, 255, 225])  # Extend hue upper limit to include orange-brown
    # lower_red2 = np.array([150, 30, 30])   # Hue: 150-180, wider range for purple-red
    # upper_red2 = np.array([180, 255, 225])

    # lower_red1 = np.array([0, 50, 50])  
    # upper_red1 = np.array([15, 255, 130])  
    # lower_red2 = np.array([160, 50, 50])  
    # upper_red2 = np.array([180, 255, 130])  

    # lower_red1 = np.array([0, 60, 20])  
    # upper_red1 = np.array([28, 255, 180])  
    # lower_red2 = np.array([170, 60, 20])  
    # upper_red2 = np.array([180, 255, 180])  

    # lower_red1 = np.array([0, 80, 30])  
    # upper_red1 = np.array([28, 255, 150])  
    # lower_red2 = np.array([170, 80, 30])  
    # upper_red2 = np.array([180, 255, 150])  

    # lower_red1 = np.array([0, 70, 25])    # Hue: 0-28°, Min Saturation: 70, Min Value: 25
    # upper_red1 = np.array([28, 255, 140]) # Max Value: 140 (excludes light beige)
    # lower_red2 = np.array([170, 70, 25])  # Hue: 170-180°, same S/V as above
    # upper_red2 = np.array([180, 255, 140])
 
    # lower_red1 = np.array([0, 50, 20])     # Min S=50 (excludes pale colors), Min V=20 (dark enough)
    # upper_red1 = np.array([30, 255, 125])   # Max V=90 (excludes washed-out colors)
    # lower_red2 = np.array([170, 50, 20])   # Same S/V thresholds
    # upper_red2 = np.array([180, 255, 125])

    lower_red1 = np.array([0, 70, 20])      # Hue range for red (0-30)
    upper_red1 = np.array([20, 255, 150])    # Reduced max hue to 20 and max value to 100 to exclude brighter colors
    lower_red2 = np.array([160, 70, 20])    # Extended lower hue to 160 for maroon
    upper_red2 = np.array([180, 255, 150])   # Reduced max value to 100

    # Create masks for both red ranges
    mask1 = cv.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel, iterations=1) #dont want to rename something new becuase lazy

    # Find contours in the maskq
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the final result
    final_mask = np.zeros_like(red_mask)

    print(f"ref: {ref_area}")

    for contour in contours:
        area = cv.contourArea(contour)
        print(f"cur area: {area}")
        
        if area > ref_area*1.5: #filter big things
            # Create a mask for the current contour
            contour_mask = np.zeros_like(red_mask)
            cv.drawContours(contour_mask, [contour], -1, 255, thickness=cv.FILLED)
            
            # Display the contour mask and the original image
            temp = cv.bitwise_and(hsv_image, hsv_image, mask=contour_mask)
            display_image = cv.cvtColor(temp, cv.COLOR_HSV2BGR)
            cv.imshow("yay or nay", display_image)
            
            # Prompt the user for input
            print("Is this a mite?")
            key = cv.waitKey(0)  # Wait for user input
            
            # If the user presses 'y', include the object in the final mask
            if key == ord('y'):
                cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)
                total_area += area
                num_objects = area/ref_area
                
                if num_objects > 2.5:
                    total_objects += num_objects
                else:
                    total_objects += 1
                print("Object included in the final mask.")
                # print(f"cur {num_objects}")
                # print(f"total {total_objects}")

            else:
                print("Object excluded from the final mask.")
            
            # Close the contour mask window
            cv.destroyWindow("yay or nay")

        elif area > ref_area*0.5: #filter small particles
            cv.drawContours(final_mask, [contour], -1, 255, thickness=cv.FILLED)
            total_area += area
            num_objects = area/ref_area
            print("added!")
            if num_objects > 1.8:
                total_objects += num_objects
            else:
                total_objects += 1
            # print(total_objects)

        else:
            print("removed!")
    
    acc_num = 103
    calc_r = total_objects

    # Display the final results
    mask_f = cv.bitwise_and(hsv_image, hsv_image, mask=final_mask)
    mark_rgb = cv.cvtColor(mask_f, cv.COLOR_HSV2BGR)
    # hsv_rgb = cv.cvtColor(mask_f, cv.COLOR_HSV2BGR)
    cv.imshow("Mites Mask", mark_rgb)
    cv.imshow("Original", normalized_img)
    print(f"Calculated num: {calc_r:.4f}, Actual num: {acc_num}")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    #take ref
    ref = reference_photo(url)

    #sticky board
    image = get_contours(url)

    ref_area = calc_ref_area(ref)

    mite_calculation(image, ref_area)