# Assignment 1
# Deng Tong     Student ID: 3160174


import cv2
import pytesseract
import thresh
from matplotlib import pyplot as plt
from pytesseract import Output
from PIL import Image
import pytesseract
from langdetect import detect_langs
import numpy as np

# Load a sample image containing text.

image1 = cv2.imread("opencv_logo.jpg")
image2 = cv2.imread("JordanPoole.png")

# Reduce the scale of the image2
scale_percent = 50
width = int(image2.shape[1] * scale_percent / 100)
height = int(image2.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Resized Image", resized_image)
cv2.waitKey()


# Pre-processing
# Apply necessary image processing techniques like thresholding, blurring, or edge detection to enhance text visibility.

# Convert the image to grayscale.
gray_image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image1",gray_image1)
cv2.imshow("Gray Image2",gray_image2)
cv2.waitKey()


# Using threshold technology(Image adaptive binarization (partition block binarization, better effect)) to enhance text visibility
binary_adaptive1 = cv2.adaptiveThreshold(
    gray_image1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow("Binary1",binary_adaptive1)

binary_adaptive2 = cv2.adaptiveThreshold(
    gray_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

cv2.imshow("Binary2",binary_adaptive2)
cv2.waitKey()

# blur
# Using a Gaussian filter, the Gaussian kernel is set to 3 pixels
gauss1 = cv2.GaussianBlur(gray_image1,(3,3),0)
gauss2 = cv2.GaussianBlur(gray_image2,(3,3),0)

# Using mean filtering, the kernel is also set to 3 pixels
median1 = cv2.medianBlur(gray_image1,3)
median2 = cv2.medianBlur(gray_image2,3)

cv2.imshow("Gauss1",gauss1)
cv2.imshow("Gauss2",gauss2)
cv2.imshow("Median1",median1)
cv2.imshow("Median2",median2)
cv2.waitKey()

# Edge Detection
# Using the Laplacian operator (detect sharp changes in edge-gradient)
laplacian1 = cv2.Laplacian(gray_image1,cv2.CV_64F)
laplacian2 = cv2.Laplacian(gray_image2,cv2.CV_64F)


# Canny edge detection (define edges as gradient Spaces)
# Gradient greater than 200 -- The change is strong enough to determine the edge
# The gradient is less than 100 -- the change is more gentle and non-edge is determined
# The gradient is in between -- undetermined to see if it is adjacent to a known edge pixel
canny1 = cv2.Canny(gray_image1,100,200)
canny2 = cv2.Canny(gray_image2,100,200)

cv2.imshow("Laplacian1",laplacian1)
cv2.imshow("Laplacian2",laplacian2)
cv2.imshow("Canny1",canny1)
cv2.imshow("Canny2",canny2)
cv2.waitKey()


# Template Matching
# Find the graph to be matched in the grayscale image and select the matching template
template1 = cv2.imread("template1.png")
template2 = cv2.imread("template2.png")

(h, w) = template1.shape[:2]
(g, k) = template2.shape[:2]

# Execute template matching
result1 = cv2.matchTemplate(resized_image, template1, cv2.TM_CCOEFF_NORMED)
result2 = cv2.matchTemplate(resized_image, template2, cv2.TM_CCOEFF_NORMED)

# Gets the minimum, maximum, and position of the match
min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)

# Use the maximum position
top_left1 = max_loc1
bottom_right1 = (top_left1[0] + w, top_left1[1] + h)
top_left2 = max_loc2
bottom_right2 = (top_left2[0] + k, top_left2[1] + g)

#Draw a rectangular box in the big picture to mark the matching position
cv2.rectangle(resized_image, top_left1, bottom_right1, (0, 255, 0), 2)
cv2.rectangle(resized_image, top_left2, bottom_right2, (0, 0 , 255), 2)

# Show matching results
cv2.imshow('Match Detected', resized_image)
cv2.waitKey()


# Text Detection
# Use OpenCV's text detection methods such as the EAST text detector, or any other OpenCV-compatible methods.
# Draw bounding boxes around detected text areas.

# Extract data with image_to_data
d = pytesseract.image_to_data(gauss1, output_type=Output.DICT)
print(d.keys())

# Return bounding box information for each detected character.
# A green rectangle is drawn around each detected character on image1.
h, w, c = image1.shape
boxes = pytesseract.image_to_boxes(image1)
for b in boxes.splitlines():
    b = b.split(' ')
    image1 = cv2.rectangle(image1, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#Displays the image with character-level bounding boxes.
cv2.imshow('text detection',image1)
cv2.waitKey()

# Loops through each detected text box and checks if the confidence level is above 60.
# Draws a blue rectangle around words with high confidence
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(image1, (x, y), (x + w, y + h), (255, 0 , 0), 2)
# Display the image
cv2.imshow('td',img)
cv2.waitKey()


# Use OCR (Optical Character Recognition) tools like Tesseract to extract text from the detected regions.
# Display the extracted text and the annotated image with bounding boxes.

# Wirte a function to read text from image
def read_text_from_image(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image. This helps in distinguishing text from the background.
  ret, thresh = cv2.threshold(gray_image, 10, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  dilation = cv2.dilate(thresh, rect_kernel, iterations = 5)

# Find contours in the dilated image, which correspond to potential text regions.
  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  image_copy = image.copy()

# For each contour, calculates the bounding rectangle and extracts the region.
# Use Tesseract OCR to extract text from the cropped region.
# Append the recognized text to document.txt.
  for contour in contours:
     x, y, w, h = cv2.boundingRect(contour)
     cropped = image_copy[y : y + h, x : x + w]
     file = open("document.txt", "a")
     text = pytesseract.image_to_string(cropped)
     file.write(text)
     file.write("\n")

  file.close()


Image = cv2.imread("opencv_logo.jpg")
read_text_from_image(Image)

# Read lines from document.txt, reverses the order, and prints them.
file = open("document.txt", "r")
lines = file.readlines()
lines.reverse()
for line in lines:
    print(line)
file.close()

# Directly perform OCR on the original image and prints the result on the console
img3 = cv2.imread("opencv_logo.jpg")
text = pytesseract.image_to_string(img3)
print(text)
