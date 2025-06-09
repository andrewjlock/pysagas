import numpy as np
import pyvista as pv

from hypervehicle.utilities import PatchTag


class CellArray:
    def __init__(self, points, dvdp, mesh):
        """A 2-D array of cell properties for use with vectorised copmutations.

        Cell properties are atomatically computed on instantiation.

        Attributes
        -----------
        """

        self.num = points.shape[0]
        self.data = np.full(
            (25, self.num), np.nan
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
            "dndp": np.moveaxis(np.einsum("ijk,lik->lij", self.dndv, self.dvdp), 2, 1),
        }

        self.A_int = np.sum(self.A)
        self.dAdp_int = np.sum(self.dAdp, axis=1)

        c_n = np.einsum("ij,ij->j", self.c, self.n)  # Shape: cells
        dcdp_n = np.einsum("ijk, kj->ij", self.dcdp, self.n)  # Shape: p x cells
        c_dndp = np.einsum("ij,kij->jk", self.c, self.dndp).T  # Shape: p x cells
        self.vol = (1 / 3) * np.sum(c_n * self.A)
        self.dvoldp = (1 / 3) * np.sum(
            dcdp_n * self.A + c_dndp * self.A + c_n * self.dAdp, axis=1
        )

        self.vol_t = 0.025
        self.vol_net = self.vol - self.A_int * self.vol_t
        self.dvol_net_dp = self.dvoldp - self.dAdp_int * self.vol_t

        i_max_l = np.argmax(self.c[0])
        i_min_l = np.argmin(self.c[0])
        i_max_w = np.argmax(self.c[1])
        i_min_w = np.argmin(self.c[1])
        i_max_h = np.argmax(self.c[2])
        i_min_h = np.argmin(self.c[2])

        max_l = self.c[0, i_max_l] - self.c[0, i_min_l]
        max_w = self.c[1, i_max_w] - self.c[1, i_min_w]
        max_h = self.c[2, i_max_h] - self.c[2, i_min_h]
        dldp = self.dcdp[:, i_max_l, 0] - self.dcdp[:, i_min_l, 0]
        dwdp = self.dcdp[:, i_max_w, 1] - self.dcdp[:, i_min_w, 1]
        dhdp = self.dcdp[:, i_max_h, 2] - self.dcdp[:, i_min_h, 2]
        l = self.c[0, i_max_l] - self.c[0, i_min_l]
        w = self.c[1, i_max_w] - self.c[1, i_min_w]
        self.AR = l / w
        self.dARdp = (dldp * w - l * dwdp) / w**2
        self.width = max_w
        self.dwidth_dp = dwdp
        self.height = max_h
        self.dheight_dp = dhdp

        # Add flow tags
        if "tag" in mesh.cell_data.keys():
            self.tag = mesh.cell_data["tag"]
        else:
            self.tag = np.full(self.num, PatchTag.FREE_STREAM.value)

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

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

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
        p = pv.Plotter()
        p.add_mesh(self.mesh, show_edges=True, scalars=scalars)
        p.show_axes()
        p.show()

    def add_flowstatevec_to_mesh(self, flowstatevec):
        """Adds flow states to mesh for plotting."""
        for key, index in flowstatevec.index.items():
            if isinstance(index, int):
                # add scalar data
                self.mesh.cell_data[key] = flowstatevec.data[index, :]
            elif isinstance(index, list):
                # add vector data
                self.mesh.cell_data.set_vectors(flowstatevec.data[index, :].T, key)
            else:
                pass

    def clear_mesh_data(self):
        """clears data attached to mesh"""
        self.mesh.clear_data()
