import numpy as np
import pyvista as pv
from typing import List

from pysagas.geometry.cell import Cell, Vector


class CellArray:
    def __init__(self, points, dvdp, mesh):
        """A 2-D array of cell properties for use with vectorised copmutations.

        Cell properties are atomatically computed on instantiation.

        Attributes
        -----------
        """

        self.num = points.shape[0]
        self.data = np.full(
            (25, self.num), np.NaN
        )  # Data except for parameter sensivities
        self.index = {
            "p0": [0, 1, 2],
            "p1": [3, 4, 5],
            "p2": [6, 7, 8],
            "A": 9,
            "n": [10, 11, 12],
            "c": [13, 14, 15],
            "id": 16,
        }
        self.mesh = mesh

        self.flow_states = []
        # Store vertices
        self.data[0:9] = points.T
        self.dvdp = dvdp

        # Calculate area
        self.data[9, :] = 0.5 * np.linalg.norm(
            np.cross((self.p1 - self.p0).T, (self.p2 - self.p0).T), axis=1
        )

        # Calc normal
        normal = np.cross((self.p1 - self.p0).T, (self.p2 - self.p0).T)
        self.data[10:13, :] = normal.T / np.linalg.norm(normal, axis=1)

        # Calc centroid
        self.data[13:16] = np.array(
            [
                (self.p0[0] + self.p1[0] + self.p2[0]) / 3,
                (self.p0[1] + self.p1[1] + self.p2[1]) / 3,
                (self.p0[2] + self.p1[2] + self.p2[2]) / 3,
            ]
        )

        self.dcdv = np.array(
            [
                [1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0, 0],
                [0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0],
                [0, 0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3],
            ]
        )

        # self.face_ids = [c._face_ids for c in cells]

        # Calc normal sensitivity
        dndv = []
        for i in range(self.num):
            dndv_ = self.calc_dndv(
                self.data[0:3, i], self.data[3:6, i], self.data[6:9, i]
            )
            dndv.append(dndv_)
        self.dndv = np.array(dndv)

        # Calc area sensitivity
        dadv = []
        for i in range(self.num):
            dadv_ = self.calc_dadv(
                self.data[0:3, i], self.data[3:6, i], self.data[6:9, i]
            )
            dadv.append(dadv_)
        self.dadv = np.array(dadv)

        self.sens = {
            "dndv": self.dndv,
            "dvdp": self.dvdp,
            "dAdp": np.einsum("ij,kij->ki", self.dadv, self.dvdp),
            "dcdp": np.einsum("ij,klj->kli", self.dcdv, self.dvdp),
            "dndp": np.moveaxis(np.einsum("ijk,lik->lij", self.dndv, self.dvdp),2,1)
        }

        self.A_int = np.sum(self.A)
        self.dAdp_int = np.sum(self.dAdp, axis=1)

    def calc_dndv(self, p0, p1, p2):
        # Use quotient rule to differentiate (a x b)/ ||a x b|| where a = p2-p0 and b=p1-p0
        # h' = (f'g - fg')/g^2
        # For this we need d/dp(||a x b||) = ((a x b)/||a x b||)*d(axb)dp
        a = p1 - p0
        b = p2 - p0
        da_dp = np.vstack([-np.eye(3), np.eye(3), np.zeros((3, 3))])
        db_dp = np.vstack([-np.eye(3), np.zeros((3, 3)), np.eye(3)])
        ab = np.cross(a, b)
        abnorm = np.linalg.norm(ab)
        dab_dp = np.cross(da_dp, b) + np.cross(a, db_dp)
        dabnorm_dp = (ab / abnorm) @ dab_dp.T
        result = (dab_dp * abnorm - np.outer(ab, dabnorm_dp).T) / abnorm**2
        return result.T

    def calc_dadv(self, p0, p1, p2):
        a = p1 - p0
        b = p2 - p0
        da_dp = np.vstack([-np.eye(3), np.eye(3), np.zeros((3, 3))])
        db_dp = np.vstack([-np.eye(3), np.zeros((3, 3)), np.eye(3)])
        ab = np.cross(a, b)
        abnorm = np.linalg.norm(ab)
        dab_dp = np.cross(da_dp, b) + np.cross(a, db_dp)
        dadv = (ab / abnorm) * dab_dp
        dadv = 0.5 * np.sum(dadv, axis=1)
        return dadv

    def __getattr__(self, name):
        if name in self.index.keys():
            return self.data[self.index[name]]
        elif name in self.sens.keys():
            return self.sens[name]
        else:
            raise ValueError(
                "Accessing CellArray value <" + name + "> that does not exist"
            )

    def set_attr(self, attr, val):
        if attr in self.index.keys():
            self.data[self.index[attr]] = val
        else:
            raise ValueError(
                "Attempting to write CellArray attribute <"
                + attr
                + "> that does not exist"
            )

    def reconstruct(self):
        cells = []
        for i in range(self.num):
            cell = Cell(
                Vector(*self.p0[:, i]),
                Vector(*self.p1[:, i]),
                Vector(*self.p2[:, i]),
                face_ids=self.face_ids[i],
            )
            cells.append(cell)
        return cells

    def plot(self, scalars=None):
        if scalars is None:
            if len(self.flow_states) > 0:
                scalars = self.flow_states[0].p
        p = pv.Plotter()
        p.add_mesh(self.mesh, show_edges=True, scalars=scalars)
        p.show_axes()
        p.show()

