import numpy as np
import cv2 as cv



pose = np.array([[1.781185e-02,  2.023573e-02,  9.996366e-01,  3.274069e+01],
                 [3.296529e-02,  9.992397e-01, -2.081509e-02, -4.366800e+00],
                 [-9.992977e-01, 3.332406e-02,  1.713123e-02,  9.059833e+01]])

R = pose[:,:3]
t = pose[:,3]


# Calculate r from R
A = (R-R.T)/2
print(A)
rmod = np.array([A[2,1], A[0,2], A[1,0]])
norm = np.linalg.norm(rmod)

print('%s = \\theta * sin(\\theta)' % norm)
# rmod == theta * sin(theta)

theta = 1.11405195588615 # Wolfram
#theta = 2.77267035122023

r = rmod/np.sin(theta)

rROD = cv.Rodrigues(R)[0][:,0]


print('calculated r: %s' % r)
print('Rodrigues r: %s' % rROD)


# Calculate R from r
theta = np.linalg.norm(r)



RmI = R-np.eye(3)


print(np.round(np.dot(R,R.T)))
print(np.round(np.dot(R.T,R)))
print(np.round(np.linalg.det(R)))


A = (R-R.T)/2
rho = np.array([A[2,1], A[0,2], A[1,0]])

s = np.linalg.norm(rho)
c = (R[0,0] + R[1,1] + R[2,2] - 1)/2

print(s)
print(c)