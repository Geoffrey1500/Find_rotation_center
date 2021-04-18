import numpy as np
import cv2


rotation_vector = np.array([[-0.07760355, -0.02770259, -0.01145906]]).T
print(rotation_vector, rotation_vector.shape)
rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
print(rotation_matrix)

tvecs = np.array([[100, 0, 0]]).T

R_and_T = np.vstack((np.hstack((rotation_matrix, tvecs)), np.array([[0, 0, 0, 1]])))
print(R_and_T)