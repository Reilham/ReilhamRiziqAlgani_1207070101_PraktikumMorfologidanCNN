import cv2
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt
# %matplotlib inline

#image = data.retina()
#image = data.astronaut()
image = imread(fname="/Kuliah Ei/Semester 6/pcd_ReilhamRiziqAlgani/Praktikum Morfologi dan CNN/Morfologi Citra/aqua2.jpg")

print(image.shape)
plt.imshow(image)
plt.show()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# defining the range of masking
blue1 = np.array([110, 50, 50])
blue2 = np.array([130, 255, 255])

# initializing the mask to be
# convoluted over input image
mask = cv2.inRange(hsv, blue1, blue2)

# passing the bitwise_and over
# each pixel convoluted
res = cv2.bitwise_and(image, image, mask=mask)

# defining the kernel i.e. Structuring element
kernel = np.ones((5, 5), np.uint8)

# defining the opening function
# over the image and structuring element
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(opening, cmap='gray')
ax[1].set_title('Citra Input 2')

import cv2
import numpy as np

# Return video from the first webcam on your computer
screenRead = cv2.VideoCapture(0)

# Loop runs if capturing has been initialized
while True:
    # Read frames from the camera
    ret, image = screenRead.read()

    # Convert to HSV color space (OpenCV reads colors as BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color for masking
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Create a mask within the specified range
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Apply bitwise AND operation to the image using the mask
    res = cv2.bitwise_and(image, image, mask=mask)

    # Define the kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological opening operation on the mask
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Show the mask and opening results in separate windows
    cv2.imshow('Mask', mask)
    cv2.imshow('Opening', opening)
    plt.show()

    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()

# Release the webcam
screenRead.release()