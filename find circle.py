from skspatial.objects import Points
from skspatial.objects import Plane
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rotation(normal_vector_, support_vector):
    a_ = np.array(normal_vector_)
    b_ = np.array(support_vector)
    theta_ = np.arccos(np.dot(a_, b_)/(np.linalg.norm(a_) * np.linalg.norm(b_)))

    rotation_axis = np.cross(a_, b_)

    q_angle = np.array([np.cos(theta_/2), np.sin(theta_/2), np.sin(theta_/2), np.sin(theta_/2)])
    q_vector = np.hstack((np.array([1]), rotation_axis))
    q = q_vector*q_angle
    q_1 = np.hstack((np.array([1]), -rotation_axis))*q_angle

    return q, q_1


def quaternion_mal(q_a, q_b):
    s = q_a[0] * q_b[0] - q_a[1] * q_b[1] - q_a[2] * q_b[2] - q_a[3] * q_b[3]
    x = q_a[0] * q_b[1] + q_a[1] * q_b[0] + q_a[2] * q_b[3] - q_a[3] * q_b[2]
    y = q_a[0] * q_b[2] - q_a[1] * q_b[3] + q_a[2] * q_b[0] + q_a[3] * q_b[1]
    z = q_a[0] * q_b[3] + q_a[1] * q_b[2] - q_a[2] * q_b[1] + q_a[3] * q_b[0]

    return np.array([s, x, y, z])


with np.load('sony_16mm.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)*30

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

cor_set_array = np.array(cor_set)
print(cor_set)
X = cor_set_array[:, 0]*-1
Y = cor_set_array[:, 1]*-1
Z = cor_set_array[:, 2]

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

points = Points(cor_set)
print(points,'chakan')
plane = Plane.best_fit(points)

point_after = []

for point in points:
    point_projected = plane.project_point(point)
    ddd = np.array(point_projected)
    point_after.append(np.array(point_projected).tolist())
    # print(ddd, "axiba", type(ddd))
# vector_projection = Vector.from_points(point, point_projected)

# print(point_projected)
point_projected_array = np.array(point_after)
print(point_projected_array, point_projected_array.shape)

X = point_projected_array[:, 0]*-1
Y = point_projected_array[:, 1]*-1
Z = point_projected_array[:, 2]

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
plt.show()

# a, b = list(normal_vector), [1, 0, 0]
# q_before, q_after = rotation(a, b)
# data_tra = np.hstack((np.zeros((cordi.shape[0], 1)), cordi))
# data_rota = quaternion_mal(q_before, quaternion_mal(data_tra.T, q_after))
# data_final = np.delete(data_rota.T, 0, axis=1)
