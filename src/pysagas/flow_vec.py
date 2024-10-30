
import numpy as np
from scipy.spatial.transform import Rotation as R

class FlowStateVec:
    def __init__(self, cells, Mach, aoa):

        self.consts = {
            "gamma": 1.4,
            "R": 287,
            "M_fs": Mach,
            "aoa": aoa,
            "cells": cells
        }
        self.data = np.full((14, cells.num), np.NaN)
        self.index = {
            "p": 0,
            "M": 1,
            "T": 2,
            "method": 3,
            "rho": 4,
            "a": 5,
            "v_mag": 6,
            "q": 7,
            "dir": [8, 9, 10],  # Flow vector direction
            "vec": [11, 12, 13],  # Flow vector
        }

    def __getattr__(self, name):
        if name in self.index.keys():
            return self.data[self.index[name]]
        elif name in self.consts.keys():
            return self.consts[name]
        else:
            raise ValueError(
                "Accessing FlowStateVec value <" + name + "> that does not exist"
            )

    def set_attr(self, attr, val):
        if attr in self.index.keys():
            self.data[self.index[attr]] = val
        else:
            raise ValueError(
                "Attempting to write FlowStateVec attribute <"
                + attr
                + "> that does not exist"
            )

    def calc_props(self):
        self.set_attr("rho", self.p / (self.R * self.T))
        self.set_attr("a", (self.gamma * self.R * self.T)**0.5)
        self.set_attr("v_mag", self.M * self.a)
        self.set_attr("q", 0.5 * self.rho * self.v_mag**2)

        # Calculate direction by projecting onto surface 
        # given by normal vector by normal
        # Note: This may fail (/0) for surfaces normal to flow
        f_fs = np.array([1, 0, 0])
        rot_mat = R.from_euler("Z", -self.aoa, degrees=1).as_matrix()
        rot_n = rot_mat@self.cells.n
        f_proj = f_fs.reshape(3,1) - np.dot(rot_n.T, f_fs) * rot_n
        f_proj = f_proj / np.linalg.norm(f_proj, axis=0)
        # f_proj = rot_mat@(f_proj.T)
        self.set_attr("dir", f_proj)
        self.set_attr("vec", self.dir * self.v_mag)
