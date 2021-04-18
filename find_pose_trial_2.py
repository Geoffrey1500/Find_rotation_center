import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

with np.load('sony_16mm.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(dist)
print(mtx)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)*30


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)*30
# print(axis)

cor_set = []

for fname in glob.glob('imgs/pose_test_2/*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        cor_set.append(tvecs.flatten().tolist())

        rotation_matrix, jacobian = cv.Rodrigues(rvecs)
        R_and_T = np.vstack((np.hstack((rotation_matrix, tvecs)), np.array([[0, 0, 0, 1]])))
        projection_mode = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]])

        # R_and_T = np.hstack((rotation_matrix, tvecs))

        # print(mtx)
        img_cor = np.linalg.multi_dot([mtx, projection_mode, R_and_T, np.array([[0], [0], [0], [1]])])
        img_cor = img_cor / img_cor[-1]

        img_cor_undistort = cv.undistortPoints(img_cor[0:-1, :], mtx, dist, None, mtx)
        print((img_cor_undistort.flatten()[0], img_cor_undistort.flatten()[1]), "坐标")

        original = np.float32([[0, 0, 0]])
        original_img, jac = cv.projectPoints(original, rvecs, tvecs, mtx, dist)

        # print(original_img.flatten())
        # print(img_cor.flatten())
        # print(img_cor_undistort.flatten())
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        cv.circle(img, (int(original_img.flatten()[0]), int(original_img.flatten()[1])), 10, (0, 0, 255), -1)
        img = draw(img, corners2, imgpts)
        cv.namedWindow("img", 0)
        cv.resizeWindow("img", 1080, 720)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

cor_set = np.array(cor_set)
print(cor_set)
X = cor_set[:, 0]*-1
Y = cor_set[:, 1]*-1
Z = cor_set[:, 2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z)

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(0, 1000)

x = [-30*9/2, -30*9/2, 30*9/2, 30*9/2]
y = [30*12/2, -30*12/2, -30*12/2, 30*12/2]
z = [0, 0, 0, 0]
verts = [list(zip(x, y, z))]
ax.add_collection3d(Poly3DCollection(verts))


# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# plt.show()
