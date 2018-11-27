'''
Vortex CAT336 Parameter
based on param_CAT336, inv_x2q MATLAB code writen by CU

Last edit date 2018.10.05
'''

from math import *
from scipy.spatial.distance import euclidean as dist
import numpy as np

#  Points from Vortex
A = [0.08, 1.709]
B = [1.012, 1.267]
C = [1.631, 4.267]
D = [5.366, 5.43]
E = [2.316, 4.99]
F = [5.89, 6.106]
G = [5.805, 4.308]
H = [4.485, 2.32]
I = [4.339, 1.862]
J = [5.1, 2.662]
K = [4.757, 2.087]
T = [4.6593, 0.2408]


class X2Q2Cyl:
    def __init__(self):
        self.OA = A[0]
        self.AL = A[1] - B[1]
        self.BL = B[0] - A[0]
        self.AB = dist(A, B)
        self.AC = dist(A, C)
        self.AD = dist(A, D)
        self.AE = dist(A, E)
        self.BC = dist(B, C)
        self.CD = dist(C, D)
        self.DE = dist(D, E)
        self.DF = dist(D, F)
        self.DI = dist(D, I)
        self.EF = dist(E, F)
        self.FG = dist(F, G)
        self.FH = dist(F, H)
        self.FI = dist(F, I)
        self.GH = dist(G, H)
        self.GI = dist(G, I)
        self.GJ = dist(G, J)
        self.HI = dist(H, I)
        self.HJ = dist(H, J)
        self.IK = dist(I, K)
        self.IT = dist(I, T)
        self.JK = dist(J, K)
        self.KT = dist(K, T)

        self.BC_min = self.BC - 0.746
        self.BC_max = self.BC + 0.5
        self.EF_min = self.EF - 1.36
        self.EF_max = self.EF + 0.108
        self.GJ_min = self.GJ
        self.GJ_max = self.GJ + 1.164

        self.theta_X0AB = cos_law(self.AB, self.BL, self.AL)

        self.theta_ADC = cos_law(self.AD, self.CD, self.AC)
        self.theta_ADE = cos_law(self.AD, self.DE, self.AE)
        self.theta_CAD = cos_law(self.AC, self.AD, self.CD)
        self.theta_DAE = cos_law(self.AD, self.AE, self.DE)
        self.theta_DIF = cos_law(self.DI, self.FI, self.DF)
        self.theta_FDI = cos_law(self.DF, self.DI, self.FI)
        self.theta_FHG = cos_law(self.FH, self.GH, self.FG)
        self.theta_FIG = cos_law(self.FI, self.GI, self.FG)
        self.theta_GFI = cos_law(self.FG, self.FI, self.GI)
        self.theta_KIT = cos_law(self.IK, self.IT, self.KT)
        self.theta_DIG = self.theta_DIF + self.theta_FIG

    def x2q(self, x, z, phi):
        #print (x,z,phi)
        tmp = (x ** 2 + z ** 2 - self.AD ** 2 - self.DI ** 2) / (2.0 * self.AD * self.DI)
        if tmp > 1:
            tmp = 1
        elif tmp < -1:
            tmp = -1
        q1 = np.zeros((1, 2))
        q2 = np.zeros((1, 2))
        q2[0][0] = acos(tmp)
        q2[0][1] = 2.0 * pi - q2[0][0]
        sc = np.zeros((2, 2))

        for i in range(2):
            a = self.AD + self.DI * cos(q2[0][i])
            b = self.DI * sin(q2[0][i])
            c = np.array([[-b, a], [a, b]])
            sc[0][i] = np.matmul(np.linalg.inv(c), np.array([[x], [z]]))[0]
            sc[1][i] = np.matmul(np.linalg.inv(c), np.array([[x], [z]]))[1]

            q1[0][i] = atan2(sc[0][i], sc[1][i])

        if self.AD * sin(q1[0][0]) > self.AD * sin(q1[0][1]):
            q3 = phi - q1[0][0] - q2[0][0]
            return [q1[0][0], q2[0][0], q3]
        else:
            q3 = phi - q1[0][1] - q2[0][1]
            return [q1[0][1], q2[0][1], q3]

    def q2cyl(self, q1, q2, q3):
        yc1 = sqrt(self.AB ** 2 + self.AC ** 2 - 2.0 * self.AB * self.AC * cos(q1 + self.theta_CAD + self.theta_X0AB));
        yc2 = sqrt(self.DE ** 2 + self.DF ** 2 - 2.0 * self.DE * self.DF * cos(
            3.0 * pi - q2 - self.theta_ADE - self.theta_FDI))

        theta_HIK = (q3 - pi) + self.theta_KIT + self.theta_DIF;
        HK = sqrt(self.HI ** 2 + self.IK ** 2 - 2 * self.HI * self.IK * cos(theta_HIK));
        theta_IHK = cos_law(self.HI, HK, self.IK);
        theta_JHK = cos_law(self.HJ, HK, self.JK);

        if (wraptopi(q3 + self.theta_KIT + self.theta_DIF) >= 0):
            theta_GHJ = pi - (self.theta_FHG + theta_IHK + theta_JHK)
        else:
            theta_GHJ = pi - self.theta_FHG - (theta_JHK - theta_IHK)

        yc3 = sqrt(self.GH ** 2 + self.HJ ** 2 - 2.0 * self.GH * self.HJ * cos(theta_GHJ))

        return yc1 - self.BC, yc2 - self.EF, yc3 - self.GJ


def cos_law(a, b, c):
    if (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b) > 1:
        res = acos(1)
    elif (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b) < -1:
        res = acos(-1)
    else:
        res = acos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))
    return res

def wraptopi(x):
    x = x - np.floor(x / (2.0 * pi)) * 2.0 * pi
    if x > pi:
        x = x - 2.0 * pi
    return x

