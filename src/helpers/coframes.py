import numpy as np
import cv2 as cv
import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

def cvtToRel(T_o_i):
    T_ip1_i = np.empty((0,4,4))
    for curr in range(len(T_o_i)-1):
        result = np.linalg.inv(T_o_i[curr+1]).dot(T_o_i[curr])
        T_ip1_i = np.append(T_ip1_i, [result], axis=0)
    return(T_ip1_i)

def cvtToAbs(T_ip1_i):
    # T_i_0[c] = T_ip1_i[c-1] * T_i_0[c-1]
    # T_0_i[c] = T_0_i[c-1] * inv(T_ip1_i[c-1])
    T_o_i = np.empty((0,4,4))
    T_o_i = np.append(T_o_i, [np.eye(4)], axis=0) # base case
    for curr in range(1,len(T_ip1_i)+1):
        result = T_o_i[curr-1].dot(np.linalg.inv(T_ip1_i[curr-1]))
        T_o_i = np.append(T_o_i, [result], axis=0)
    return (T_o_i)

def getRT_sd_ss2d(T_s_d):
    R_s_d, t_s_s2d = T_s_d[:, :3, :3], T_s_d[:, :3, 3:4]
    return (R_s_d, t_s_s2d)

def getT_s_d(R_s_d, t_ss2d):
    T_s_d = np.zeros((len(R_s_d), 4, 4))
    T_s_d[:, :3, :3] = R_s_d
    T_s_d[:, :3, 3] = t_ss2d
    T_s_d[:, 3, 3] = np.ones(len(R_s_d))
    return (T_s_d)

def getXYZ_ss2d(t_s_s2d):
    x_s_s2d, y_s_s2d, z_s_s2d = t_s_s2d[:, :, 0].T
    return (x_s_s2d, y_s_s2d, z_s_s2d)

def cvtToRpy_sd(R_s_d):
    rpy_s_d = np.empty((0,1,3))
    for curr in range(len(R_s_d)):
        result = cv.Rodrigues(np.linalg.inv(R_s_d[curr]))[0].T
        rpy_s_d = np.append(rpy_s_d, [result], axis=0)
    return (rpy_s_d)

def cvtToDcm_sd(rpy_s_d):
    dcm_s_d = np.empty((0,3,3))
    for curr in range(len(rpy_s_d)):
        result = np.linalg.inv(cv.Rodrigues(rpy_s_d[curr])[0])
        dcm_s_d = np.append(dcm_s_d, [result], axis=0)
    return (dcm_s_d)


# https://en.wikipedia.org/wiki/Spherical_coordinate_system
def cart2sph(xyz):
    rtp = np.zeros(xyz.shape)
    # R = sqrt(x^2 + y^2 + z^2)
    rtp[:, 0] = np.sqrt(np.sum(xyz**2, axis=1))
    # theta (inclination) = acos(z/R)
    # Right handed rotation about y from z
    rtp[:, 1] = np.arccos(xyz[:,2]/rtp[:,0])
    # phi (aximuth) = atan2(y,x)
    # Right handed rotation about z from x
    rtp[:, 2] = np.arctan2(xyz[:,1], xyz[:,0])
    return rtp

def sph2cart(rtp):
    xyz = np.zeros(rtp.shape)
    # X = R * sin (theta) * cos (phi)
    xyz[:, 0] = rtp[:,0] * np.sin(rtp[:,1]) * np.cos(rtp[:,2])
    # Y = R * sin (theta) * sin (phi)
    xyz[:, 1] = rtp[:,0] * np.sin(rtp[:,1]) * np.sin(rtp[:,2])
    # Z = R * cos (theta)
    xyz[:, 2] = rtp[:,0] * np.cos(rtp[:,1])
    return xyz

# def plotBirdsEye(T_o_i):
#     _, t_o_o2i = getRT_sd_ss2d(T_o_i)
#     x, _, z = getXYZ_ss2d(t_o_o2i)
#     plt.plot(x, z)