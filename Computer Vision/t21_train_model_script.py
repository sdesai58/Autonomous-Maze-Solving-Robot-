# #!/usr/bin/env python3
# import cv2
# import sys
# import csv
# import time
# import math
# import numpy as np
# import random
# import pickle

# # Load training images and labels

# imageDirectory = './2024Simgs/'

# with open(imageDirectory + 'labels.txt', 'r') as f:
#     reader = csv.reader(f)
#     lines = list(reader)

# def image_process(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     upper_red = np.array([25, 255, 255])
#     lower_red = np.array([0, 20, 20])
#     upper_green = np.array([90, 255, 255])
#     lower_green = np.array([50, 100, 50])
#     upper_blue = np.array([130, 255, 255])
#     lower_blue = np.array([90, 40, 30])

#     thresh_red = cv2.inRange(img, lower_red, upper_red)
#     thresh_green = cv2.inRange(img, lower_green, upper_green)
#     thresh_blue = cv2.inRange(img, lower_blue, upper_blue)

#     mask = thresh_red | thresh_green | thresh_blue

#     kernel = np.ones((5, 5), np.uint8)
#     closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

#     contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         return np.zeros((40, 40), dtype=np.uint8)

#     # Assuming you want to find the contour with the largest area
#     max_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(max_contour)
#     cropped_img = opening[y:y+h, x:x+w]

#     # Resize the cropped image to a fixed shape
#     cropped_img = cv2.resize(cropped_img, (40, 40))

#     return cropped_img



# # Randomly choose train and test data (80% train, 20% test)
# random.shuffle(lines)
# train_lines = lines[:math.floor(len(lines)*0.8)][:]
# test_lines = lines[math.floor(len(lines)*0.2):][:]

# # Load and preprocess train images
# train_in = np.array([cv2.imread(imageDirectory+line[0]+".png") for line in train_lines])
# train_out = np.array([image_process(img) for img in train_in])
# train_data = train_out.flatten().reshape(len(train_lines), -1).astype(np.float32)
# train_labels = np.array([int(line[1]) for line in train_lines])

# # Load and preprocess test images
# test_in = np.array([cv2.imread(imageDirectory+line[0]+".png") for line in test_lines])
# test_out = np.array([image_process(img) for img in test_in])
# test_data = test_out.flatten().reshape(len(test_lines), -1).astype(np.float32)
# test_labels = np.array([int(line[1]) for line in test_lines])

# # Train KNN classifier
# knn = cv2.ml.KNearest_create()
# knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# # Test classifier
# correct = 0
# confusion_matrix = np.zeros((6, 6))
# k = 1

# for i in range(len(test_lines)):
#     ret, _, _, _ = knn.findNearest(test_data[i:i+1], k)
#     ret = int(ret)
#     if ret == test_labels[i]:
#         correct += 1
#     confusion_matrix[test_labels[i], ret] += 1

# accuracy = correct / len(test_lines)
# print("Accuracy:", accuracy)
# print("Confusion Matrix:")
# print(confusion_matrix)

# # saving the model
# knn.save("knn_model.xml")

import sys
import csv
import cv2
import time
import math
import numpy as np
import random
import pickle
import sklearn 
from sklearn.neighbors import KNeighborsClassifier

# Load training images and labels
imageDirectory = './2024Simgs/'

with open(imageDirectory + 'labels.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

def image_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_red = np.array([25, 255, 255])
    lower_red = np.array([0, 20, 20])
    upper_green = np.array([90, 255, 255])
    lower_green = np.array([50, 100, 50])
    upper_blue = np.array([130, 255, 255])
    lower_blue = np.array([90, 40, 30])

    thresh_red = cv2.inRange(img, lower_red, upper_red)
    thresh_green = cv2.inRange(img, lower_green, upper_green)
    thresh_blue = cv2.inRange(img, lower_blue, upper_blue)

    mask = thresh_red | thresh_green | thresh_blue

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((40, 40), dtype=np.uint8)

    # Assuming you want to find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_img = opening[y:y+h, x:x+w]

    # Resize the cropped image to a fixed shape
    cropped_img = cv2.resize(cropped_img, (40, 40))

    return cropped_img

# Randomly choose train and test data (80% train, 20% test)
random.shuffle(lines)
train_lines = lines[:math.floor(len(lines)*0.8)][:]
test_lines = lines[math.floor(len(lines)*0.2):][:]

# Load and preprocess train images
train_in = np.array([cv2.imread(imageDirectory+line[0]+".png") for line in train_lines])
train_out = np.array([image_process(img) for img in train_in])
train_data = train_out.flatten().reshape(len(train_lines), -1).astype(np.float32)
train_labels = np.array([int(line[1]) for line in train_lines])

# Load and preprocess test images
test_in = np.array([cv2.imread(imageDirectory+line[0]+".png") for line in test_lines])
test_out = np.array([image_process(img) for img in test_in])
test_data = test_out.flatten().reshape(len(test_lines), -1).astype(np.float32)
test_labels = np.array([int(line[1]) for line in test_lines])

# Train KNN classifier using scikit-learn
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_labels)

# Test classifier
correct = 0
confusion_matrix = np.zeros((6, 6))

for i in range(len(test_lines)):
    ret = knn.predict(test_data[i:i+1])
    ret = int(ret)
    if ret == test_labels[i]:
        correct += 1
    confusion_matrix[test_labels[i], ret] += 1

accuracy = correct / len(test_lines)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix)

# saving the model
pickle.dump(knn, open("knn_model.pkl", "wb"))



