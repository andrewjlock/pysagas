import numpy as np
from scipy.spatial.transform import Rotation as R
from hypervehicle.utilities import PatchTag
from hypervehicle.geometry import Vector3

# from examples.wedge.sensitivity import freestream


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

    @property
    def Cp(self):
        return self.R * self._gamma / (self._gamma - 1)


class FlowState(GasState):
    """An ideal gas state defined by Mach number, pressure and
    temperature, with a flow direction.
    """

    def __init__(
        self,
        mach: float,
        pressure: float,
        temperature: float,
        direction=None,
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
            if isinstance(direction, Vector3):
                self.direction = np.array(
                    [direction.unit.x, direction.unit.y, direction.unit.z]
                )
            else:
                self.direction = direction.unit
        else:
            # Use AoA to calculate direction
            rot_mat = R.from_euler("ZYX", [0, -aoa, 0], degrees=1).as_matrix()
            vec = rot_mat @ np.array([-1, 0, 0])
            self.direction = vec / np.linalg.norm(
                vec
            )  # Shouldn't be needed but doesn't hurt

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
        # return self.direction * self.v
        return np.array(
            [
                self.direction[0] * self.v,
                self.direction[1] * self.v,
                self.direction[2] * self.v,
            ]
        )

    @property
    def aoa(self):
        aoa = np.rad2deg(np.arctan(self.vec[2] / self.vec[0]))
        return aoa


class FlowStateVec:
    def __init__(self, cells, Mach, aoa):

        self.consts = {"gamma": 1.4, "R": 287, "M_fs": Mach, "aoa": aoa, "cells": cells}
        self.data = np.full((15, cells.num), np.nan)
        self.index = {
            "p": 0,
            "M": 1,
            "T": 2,
            "method": 3,
            "rho": 4,
            "a": 5,
            "v_mag": 6,
            "q": 7,
            "direction": [8, 9, 10],  # Flow vector direction
            "vec": [11, 12, 13],  # Flow vector
        }
        self.p_sens = []

    def __getattr__(self, name):
        # Prevent Python internals from breaking things
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{name} not found")

        index = self.__dict__.get("index")
        if index is not None and name in index:
            return self.data[self.index[name]]

        consts = self.__dict__.get("consts")
        if consts is not None and name in consts:
            return consts[name]

        raise ValueError(
            "Accessing FlowStateVec value <" + name + "> that does not exist"
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
                "Attempting to write FlowStateVec attribute <"
                + attr
                + "> that does not exist"
            )

    def calc_props(self):
        self.set_attr("rho", self.p / (self.R * self.T))
        self.set_attr("a", (self.gamma * self.R * self.T) ** 0.5)
        self.set_attr("v_mag", self.M * self.a)
        self.set_attr("q", 0.5 * self.rho * self.v_mag**2)

        # Calculate direction by projecting onto surface
        # given by normal vector by normal
        f_fs = np.array([-1, 0, 0])
        rot_mat = R.from_euler("Y", -self.aoa, degrees=1).as_matrix()
        f_fs = rot_mat @ f_fs
        f_proj = f_fs.reshape(3, 1) - np.dot(self.cells.n.T, f_fs) * self.cells.n
        # We find instances where the surfaces are normal to the flow and set the
        # flow direction as free strepes
        singulars = np.where(np.linalg.norm(f_proj, axis=0) == 0)[0]
        f_proj[:, singulars] = np.repeat(f_fs.reshape(3, -1), len(singulars), axis=1)
        f_proj = f_proj / np.linalg.norm(f_proj, axis=0)
        self.set_attr("direction", f_proj)
        self.set_attr("vec", self.direction * self.v_mag)


class InFlowStateVec:
    def __init__(self, cells, freestream, eng_outflow=None):

        if any(cells.tag == PatchTag.NOZZLE) and eng_outflow is None:
            raise Exception("eng_outflow cannot be None")

        if eng_outflow is None:
            eng_outflow = freestream

        self.freestream = freestream
        self.eng_outflow = eng_outflow
        self.cells = cells

        ind_nozzle = self.cells.tag == PatchTag.NOZZLE.value
        self.P = np.where(ind_nozzle, eng_outflow.P, freestream.P)
        self.M = np.where(ind_nozzle, eng_outflow.M, freestream.M)
        self.T = np.where(ind_nozzle, eng_outflow.T, freestream.T)
        self.rho = np.where(ind_nozzle, eng_outflow.rho, freestream.rho)
        self.a = np.where(ind_nozzle, eng_outflow.a, freestream.a)
        self.v = np.where(ind_nozzle, eng_outflow.v, freestream.v)
        self.q = np.where(ind_nozzle, eng_outflow.q, freestream.q)
        self.direction = np.where(
            ind_nozzle[:, None], eng_outflow.direction, freestream.direction
        ).T
        self.vec = np.where(ind_nozzle[:, None], eng_outflow.vec, freestream.vec).T
        self.aoa = np.where(ind_nozzle, eng_outflow.aoa, freestream.aoa)
        self.gamma = np.where(ind_nozzle, eng_outflow.gamma, freestream.gamma)
        self.R = np.where(ind_nozzle, eng_outflow.R, freestream.R)


class FlowResults:
    """A class containing the aerodynamic force and moment
    information for a single flow condition.

    Attributes
    ----------
    freestream : FlowState
        The freestream flow state.

    net_force : pd.DataFrame
        The net force in cartesian coordinate frame (x,y,z).

    m_sense : pd.DataFrame
        The net moment in cartesian coordinate frame (x,y,z).
    """

    def __init__(
        self,
        freestream: FlowState,
        net_force: Vector3,
        net_moment: Vector3,
    ) -> None:
        self.freestream = freestream
        self.net_force = net_force
        self.net_moment = net_moment

        # # Calculate angle of attack
        # self.aoa = np.rad2deg(
        #     np.arctan(freestream.direction.y / freestream.direction.x)
        # )

    def __str__(self) -> str:
        return f"Net force = {self.net_force} N"

    def __repr__(self) -> str:
        return self.__str__()
