import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt
import imutils

# Function to process the image and recognize the number plate
def recognize_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop over contours to find the best approximate contour
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # We want a quadrilateral
            location = approx
            break

    if location is None:
        print("No license plate detected.")
        return

    # Create a mask
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], -1, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Find the coordinates of the four corners of the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if result:
        text = result[0][-2]  # Extract the text from the result
        print(f"Detected License Plate Number: {text}")

        # Draw rectangle and put text on the original image
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_with_text = cv2.putText(img, text=text, org=(y1, x1 + 30),
                                     fontFace=font, fontScale=1, color=(0, 255, 0),
                                     thickness=2, lineType=cv2.LINE_AA)
        img_with_rectangle = cv2.rectangle(img_with_text, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

        # Show the result
        plt.imshow(cv2.cvtColor(img_with_rectangle, cv2.COLOR_BGR2RGB))
        plt.title('Final Result with Detected Text')
        plt.axis('off')
        plt.show()
    else:
        print("No text detected.")

# Call the function with the path to your number plate image
recognize_number_plate('frame.jpg')  # Update with your image path
