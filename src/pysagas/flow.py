import numpy as np
from scipy.spatial.transform import Rotation as R

class GasState:
    """An ideal gas state defined by Mach number, pressure and
    temperature.
    """

    gamma = 1.4
    R = 287  # J/kg·K.3

    def __init__(
        self, mach: float, pressure: float, temperature: float, gamma: float = 1.4
    ) -> None:
        """Define a new gas state.

        Parameters
        -----------
        mach : float
            The flow Mach number.

        pressure : float
            The flow pressure (Pa).

        temperature : float
            The flow temperature (K).

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.
        """
        # Assign properties
        self._T = temperature
        self._P = pressure
        self._M = mach
        self._gamma = gamma

    def __str__(self) -> str:
        return f"Mach {self.M} flow condition with P = {self.P}, T = {self.T}."

    def __repr__(self) -> str:
        return f"Flow(M={self.M}, P={self.P}, T={self.T})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GasState):
            raise Exception(f"Cannot compare {type(other)} to GasState.")
        return (
            (self._T == other._T)
            & (self._P == other._P)
            & (self._M == other._M)
            & (self._gamma == other._gamma)
        )

    @property
    def T(self):
        return self._T

    @property
    def P(self):
        return self._P

    @property
    def M(self):
        return self._M

    @property
    def a(self):
        return (self.gamma * self.R * self.T) ** 0.5

    @property
    def rho(self):
        return self.P / (self.R * self.T)

    @property
    def v(self):
        return self.M * self.a

    @property
    def q(self):
        return 0.5 * self.rho * self.v**2

    @property
    def gamma(self):
        return self._gamma


class FlowState(GasState):
    """An ideal gas state defined by Mach number, pressure and
    temperature, with a flow direction.
    """

    def __init__(
        self,
        mach: float,
        pressure: float,
        temperature: float,
        direction = None,
        aoa: float = 0.0,
        gamma: float = 1.4,
    ) -> None:
        """Define a new flow state.

        Parameters
        -----------
        mach : float
            The flow Mach number.

        pressure : float
            The flow pressure (Pa).

        temperature : float
            The flow temperature (K).

        direction : Vector, optional
            The direction vector of the flow. The default is Vector(1,0,0).

        aoa : float, optional
            The angle of attack of the flow. The default is 0.0 (specified in
            degrees).

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.
        """
        super().__init__(mach, pressure, temperature, gamma)
        if direction:
            # Use direction provided
            self.direction = direction.unit
        else:
            # Use AoA to calculate direction
            vec = direction = np.array([1, 1 * np.tan(np.deg2rad(aoa)), 0])
            self.direction = vec / np.linalg.norm(vec)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowState):
            raise Exception(f"Cannot compare {type(other)} to FlowState.")
        same_gs = super().__eq__(other)
        return same_gs & (self.direction == other.direction)

    @property
    def vx(self):
        return self.Vector[0]

    @property
    def vy(self):
        return self.Vector[1]

    @property
    def vz(self):
        return self.Vector[2]

    @property
    def vec(self):
        return self.Vector

    @property
    def Vector(self):
        return self.direction * self.v

    @property
    def aoa(self):
        aoa = np.rad2deg(np.arctan(self.vec[1] / self.vec[0]))
        return round(aoa, 6)


if __name__ == "__main__":
    flow = FlowState(6, 700, 70)

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
