import numpy as np
from typing import List
from pysagas.geometry.cell import Cell, Vector
import autograd.numpy as npa
from autograd import elementwise_grad as egrad
from autograd import jacobian, grad


class CellArray:
    def __init__(self, cells: List[Cell]):
        """A 2-D array of cell properties for use with vectorised copmutations.

        Cell properties are atomatically computed on instantiation.

        Attributes
        -----------
        """

        self.num = len(cells)
        self.data = np.full(
            (21, self.num), np.NaN
        )  # Data except for parameter sensivities
        self.index = {
            "p0": [0, 1, 2],
            "p1": [3, 4, 5],
            "p2": [6, 7, 8],
            "A": 9,
            "n": [10, 11, 12],
            "c": [13, 14, 15],
            "id": 16,
            "pressure": 17,
            "Mach": 18,
            "temperature": 19,
            "method": 20,
        }

        # Store vertices
        self.data[0:9, :] = np.array(
            [np.concatenate([c.p0.vec, c.p1.vec, c.p2.vec]) for c in cells]
        ).T

        # Calculate area
        self.data[9, :] = 0.5 * np.linalg.norm(
            np.cross((self.p1 - self.p0).T, (self.p2 - self.p0).T), axis=1
        )

        # Calc normal
        normal = np.cross((self.p2 - self.p0).T, (self.p1 - self.p0).T)
        self.data[10:13, :] = normal.T / np.linalg.norm(normal, axis=1)

        # Calc centroid
        self.data[13:16] = np.array(
            [
                (self.p0[0] + self.p1[0] + self.p2[0]) / 3,
                (self.p0[1] + self.p1[1] + self.p2[1]) / 3,
                (self.p0[2] + self.p1[2] + self.p2[2]) / 3,
            ]
        )

        self.face_ids = [c._face_ids for c in cells]

        # Get dndp
        # Note: Ideally we vectorise this computation for speed.
        # For now I will just copy values from cells.
        # Can we just cheat and use an autograd package? (I.e. Princeton autograd)
        self.dndv = np.array([c.dndv.T for c in cells]).T

        # Get dvdp
        # As above, ideally we vectorise this computation but for now will simply copy
        self.dvdp = np.array([c.dvdp.T for c in cells]).T

        # Get dadp
        self.dAdp = np.array([c.dAdp.T for c in cells]).T

        # Get dcdp
        self.dcdp = np.array([c.dcdp.T for c in cells]).T

        self.sens = {
            "dndv": self.dndv,
            "dvdp": self.dvdp,
            "dAdp": self.dAdp,
            "dcdp": self.dcdp,
        }


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
                "Attempting to write CellArray attribute <" + attr + "> that does not exist"
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
            # cell.dndv = self.dndv[:, :, i]
            # cell.dvdp = self.dvdp[:, :, i]
            # cell.dAdp = self.dAdp[:, :, i]
            # cell.dcdp = self.dcdp[:, :, i]
            cells.append(cell)
        return cells
