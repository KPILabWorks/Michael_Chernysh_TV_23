import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:\\Users\\cherm\\PycharmProjects\\DataAnal\\lab5\\data\\house.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.axis('off')
plt.show()
