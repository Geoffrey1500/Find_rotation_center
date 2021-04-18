import numpy as np
import cv2

rotation_matrix = np.zeros(shape=(3,3))
cv2.Rodrigues((1, 0, 0), rotation_matrix)
#Apply rotation matrix to point
original_point = np.array([[1],[0],[0]])
rotated_point = rotation_matrix*original_point
print(rotated_point)