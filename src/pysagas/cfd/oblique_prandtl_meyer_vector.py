import numpy as np
from typing import Optional, Union
from pysagas.flow import FlowStateVec, FlowState, InFlowStateVec

from hypervehicle.utilities import PatchTag


class OPMVec:
    """Oblique shock thoery and Prandtl-Meyer expansion theory flow solver using
    vectorised computations.

    This implementation uses oblique shock theory for flow-facing cell
    elements, and Prandtl-Meyer expansion theory for rearward-facing elements.

    Extended Summary
    ----------------
    Data attribute 'method' refers to which method was used for a particular
    cell, according to:
        - -1 : invalid / skipped (eg. 90 degree face)
        - 0 : parallel face, do nothing
        - 1 : Prandlt-Meyer
        - 2 : normal shock
        - 3 : oblique shock
    """

    PM_ANGLE_THRESHOLD = -20  # degrees
    EPS = 1e-15

    def solve(
        self,
        cells,
        freestream: Union[FlowState, InFlowStateVec]=None,
        cog=np.array([0, 0, 0]),
        A_ref: float = 1,
        c_ref: float = 1,
    ):

        if isinstance(freestream, FlowState):
            flow = freestream
            inflow = InFlowStateVec(cells, freestream)
        elif isinstance(freestream, InFlowStateVec):
            flow = freestream.freestream
            inflow = freestream

        M2 = np.full(cells.num, float(flow.M))
        p2 = np.full(cells.num, 0.0)
        T2 = np.full(cells.num, float(flow.T))
        method = np.full(cells.num, -1)

        if inflow.direction.ndim == 1:
            dot4theta = np.einsum('i,ij->j', -inflow.direction, cells.n)
        else:
            dot4theta = np.einsum('ij,ij->j', -inflow.direction, cells.n)

        theta = np.pi / 2 - np.arccos(dot4theta
            / (np.linalg.norm(cells.n, axis=0) * np.linalg.norm(inflow.direction, axis=0))
        )
        theta = np.where(abs(theta) < self.EPS, 0.0, theta)

        r = cells.c - cog.reshape(3, 1)
        beta_max = OPMVec.beta_max(M=inflow.M, gamma=inflow.gamma)
        theta_max = OPMVec.theta_from_beta(M1=inflow.M, beta=beta_max, gamma=inflow.gamma)

        # INLET and OUTLET cells aerodynamics are not computed
        dont_calc = (cells.tag == PatchTag.INLET.value) | (cells.tag == PatchTag.OUTLET.value)

        bad_idx = np.where((theta < np.deg2rad(self.PM_ANGLE_THRESHOLD)) & (~dont_calc)) # "bad" cells - exceed max value for P-M turning angle
        pm_idx = np.where((np.deg2rad(self.PM_ANGLE_THRESHOLD) < theta) & (theta < 0) & (~dont_calc)) # P-M solver
        parallel_idx = np.where((theta == 0) & (~dont_calc)) # Cells parallel to the incoming flow - no aerodynamic forces
        oblique_idx = np.where((theta_max > theta) & (theta > 0) & (~dont_calc)) # Oblique shock
        normal_idx = np.where((theta > theta_max) & (~dont_calc)) # Normal Shock

        if pm_idx[0].size > 0:
            M2[pm_idx], p2[pm_idx], T2[pm_idx] = self._solve_pm(
                abs(theta[pm_idx]), inflow.M[pm_idx], inflow.P[pm_idx], inflow.T[pm_idx], inflow.gamma[pm_idx])

        if parallel_idx[0].size > 0:
            M2[parallel_idx], p2[parallel_idx], T2[parallel_idx] = (inflow.M[parallel_idx], inflow.P[parallel_idx],
                                                                    inflow.T[parallel_idx])
        if oblique_idx[0].size > 0:
            M2[oblique_idx], p2[oblique_idx], T2[oblique_idx] = self._solve_oblique(
                abs(theta[oblique_idx]), inflow.M[oblique_idx], inflow.P[oblique_idx], inflow.T[oblique_idx],
                inflow.gamma[oblique_idx])

        if normal_idx[0].size > 0:
            M2[normal_idx], p2[normal_idx], T2[normal_idx] = self._solve_normal(
                inflow.M[normal_idx], inflow.P[normal_idx], inflow.T[normal_idx], inflow.gamma[normal_idx])


        method[bad_idx] = -1
        method[pm_idx] = 1
        method[parallel_idx] = 0
        method[oblique_idx] = 3
        method[normal_idx] = 2

        flow_state = FlowStateVec(cells, flow.M, flow.aoa)
        flow_state.set_attr("p", p2)
        flow_state.set_attr("M", M2)
        flow_state.set_attr("T", T2)
        flow_state.set_attr("method", method)
        flow_state.calc_props()

        F = -cells.n * p2 * cells.A
        force = np.sum(F, axis=1)
        moment = np.sum(np.cross(r.T, F.T).T, axis=1)
        C_force = force / (flow.q * A_ref)
        C_moment = moment / (flow.q * A_ref * c_ref)

        bad = len(bad_idx)
        if bad / cells.num > 0.25:
            print(
                f"WARNING: {100*bad/cells.num:.2f}% of cells were not "
                "solved due to PM threshold."
            )

        result = {
            "F": force,
            "M": moment,
            "CF": C_force,
            "CM": C_moment,
            "freestream": flow,
        }

        return result, flow_state

    @staticmethod
    def pm(M: float, gamma: float = 1.4):
        """Solves the Prandtl-Meyer function and returns the Prandtl-Meyer angle
        in radians."""
        v = ((gamma + 1) / (gamma - 1)) ** 0.5 * np.arctan(
            ((M**2 - 1) * (gamma - 1) / (gamma + 1)) ** 0.5
        ) - np.arctan((M**2 - 1) ** 0.5)
        return v

    @staticmethod
    def inv_pm(angle: float, gamma: float = 1.4):
        """Solves the inverse Prandtl-Meyer function using a bisection algorithm."""
        func = lambda M: OPMVec.pm(M, gamma) - angle
        # Use a manual vectorised version of bisection method here
        max_step = 200
        x_0 = np.full(len(angle), 1)
        x_1 = np.full(len(angle), 42)
        for step in range(max_step):
            x_mid = (x_0 + x_1) / 2.0
            f_0 = func(x_0)
            f_1 = func(x_1)
            f_mid = func(x_mid)
            x_0 = np.where(np.sign(f_mid) == np.sign(f_0), x_mid, x_0)
            x_1 = np.where(np.sign(f_mid) == np.sign(f_1), x_mid, x_1)
            error_max = np.amax(np.abs(x_1 - x_0))
            if error_max < 1e-6:
                break
        return x_1
        # pm = bisect(func, 1.0, 42.0)
        # return pm

    @staticmethod
    def _solve_pm(
        theta: float, M1: float, p1: float = 1.0, T1: float = 1.0, gamma: float = 1.4
    ):
        # Solve for M2
        v_M1 = OPMVec.pm(M=M1, gamma=gamma)
        v_M2 = theta + v_M1
        if len(v_M2) > 0:
            M2 = OPMVec.inv_pm(angle=v_M2, gamma=gamma)
        else:
            M2 = M1

        a = (gamma - 1) / 2
        n = 1 + a * M1**2
        d = 1 + a * M2**2

        p2 = p1 * (n / d) ** (gamma / (gamma - 1))

        T2 = T1 * (n / d)

        return M2, p2, T2

    @staticmethod
    def _solve_oblique(
        theta: float, M1: float, p1: float = 1.0, T1: float = 1.0, gamma: float = 1.4
    ):
        """Solves the flow using oblique shock theory.

        Parameters
        ----------
        theta : float
            The flow deflection angle, specified in radians.

        M1 : float
            The pre-shock Mach number.

        p1 : float, optional
            The pre-shock pressure (Pa). The default is 1.0.

        T1 : float, optional
            The pre-expansion temperature (K). The default is 1.0.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        M2 : float
            The post-shock Mach number.

        p2 : float
            The post-shock pressure (Pa).

        T2 : float
            The post-shock temperature (K).
        """
        # Calculate angle and ratios
        beta = OPMVec.oblique_beta(M1, theta, gamma)
        p2_p1 = OPMVec.oblique_p2_p1(M1, beta, gamma)
        rho2_rho1 = OPMVec.oblique_rho2_rho1(M1, beta, gamma)
        T2_T1 = p2_p1 / rho2_rho1

        # Calculate properties
        M2 = OPMVec.oblique_M2(M1, beta, theta, gamma)
        p2 = p2_p1 * p1
        T2 = T1 * T2_T1

        return M2, p2, T2

    @staticmethod
    def _solve_normal(M1: float, p1: float = 1.0, T1: float = 1.0, gamma: float = 1.4):
        """Solves the flow using normal shock theory.

        Parameters
        ----------
        M1 : float
            The pre-shock Mach number.

        p1 : float, optional
            The pre-shock pressure (Pa). The default is 1.0.

        T1 : float, optional
            The pre-expansion temperature (K). The default is 1.0.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        M2 : float
            The post-shock Mach number.

        p2 : float
            The post-shock pressure (Pa).

        T2 : float
            The post-shock temperature (K).
        """
        rho2_rho1 = (gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2)
        p2_p1 = 1 + 2 * gamma * (M1**2 - 1) / (gamma + 1)

        M2 = ((1 + M1**2 * (gamma - 1) / 2) / (gamma * M1**2 - (gamma - 1) / 2)) ** 0.5
        p2 = p1 * p2_p1
        T2 = T1 * p2_p1 / rho2_rho1

        return M2, p2, T2

    @staticmethod
    def oblique_p2_p1(M1: float, beta: float, gamma: float = 1.4):
        """Returns the pressure ratio p2/p1 across an oblique shock.

        Parameters
        -----------
        M1 : float
            The pre-shock Mach number.

        beta : float
            The shock angle specified in radians.

        Returns
        --------
        p2/p1 : float
            The pressure ratio across the shock.

        References
        ----------
        Peter Jacobs
        """
        M1n = M1 * abs(np.sin(beta))
        p2p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1n**2 - 1.0)
        return p2p1

    @staticmethod
    def oblique_beta(
        M1: float, theta: float, gamma: float = 1.4, tolerance: float = 1.0e-6
    ):
        """Calculates the oblique shock angle using a recursive algorithm.

        Parameters
        ----------
        M1 : float
            The pre-shock Mach number.

        theta : float
            The flow deflection angle, specified in radians.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        References
        ----------
        Peter Jacobs
        """
        func = lambda beta: OPMVec.theta_from_beta(M1, beta, gamma) - theta

        # Initialise
        sign_beta = np.sign(theta)
        theta = abs(theta)
        b1 = np.arcsin(1.0 / M1) * 1
        b2 = OPMVec.beta_max(M=M1, gamma=gamma)
        # b2 = np.arcsin(1.0 / M1) * 3

        # Check f1
        f1 = func(b1)
        if np.max(np.abs(f1)) < tolerance:
            return sign_beta * b1

        # Check f2
        f2 = func(b2)
        if np.max(np.abs(f2)) < tolerance:
            return sign_beta * b2

        # Instead of secand method solve again with bisection method
        max_step = 200
        x_0 = np.full(len(theta), b1)
        x_1 = np.full(len(theta), b2)
        for step in range(max_step):
            x_mid = (x_0 + x_1) / 2.0
            f_0 = func(x_0)
            f_1 = func(x_1)
            f_mid = func(x_mid)
            x_0 = np.where(np.sign(f_mid) == np.sign(f_0), x_mid, x_0)
            x_1 = np.where(np.sign(f_mid) == np.sign(f_1), x_mid, x_1)
            error_max = np.amax(np.abs(x_1 - x_0))
            if error_max < 1e-6:
                break
        beta = sign_beta * x_1

        return beta

    @staticmethod
    def theta_from_beta(M1: float, beta: float, gamma: float = 1.4):
        """Calculates the flow deflection angle from the oblique shock angle.

        Parameters
        ----------
        M1 : float
            The pre-expansion Mach number.

        beta : float
            The shock angle specified in radians.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        theta : float
            The deflection angle, specified in radians.

        References
        ----------
        Peter Jacobs
        """
        M1n = M1 * abs(np.sin(beta))
        t1 = 2.0 / np.tan(beta) * (M1n**2 - 1)
        t2 = M1**2 * (gamma + np.cos(2 * beta)) + 2
        theta = np.arctan(t1 / t2)
        return theta

    @staticmethod
    def oblique_M2(M1: float, beta: float, theta: float, gamma: float = 1.4):
        """Calculates the Mach number following an oblique shock.

        Parameters
        ----------
        M1 : float
            The pre-expansion Mach number.

        beta : float
            The shock angle specified in radians.

        theta : float
            The deflection angle, specified in radians.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        M2 : float
            The post-shock Mach number.

        References
        ----------
        Peter Jacobs
        """
        M1n = M1 * abs(np.sin(beta))
        a = 1 + (gamma - 1) * 0.5 * M1n**2
        b = gamma * M1n**2 - (gamma - 1) * 0.5
        M2 = (a / b / (np.sin(beta - theta)) ** 2) ** 0.5
        return M2

    @staticmethod
    def oblique_T2_T1(M1: float, beta: float, gamma: float = 1.4):
        """Returns the temperature ratio T2/T1 across an oblique shock.

        Parameters
        -----------
        M1 : float
            The pre-shock Mach number.

        beta : float
            The shock angle specified in radians.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        T2/T1 : float
            The temperature ratio across the shock.

        References
        ----------
        Peter Jacobs
        """
        T2_T1 = OPMVec.oblique_p2_p1(M1, beta, gamma) / OPMVec.oblique_rho2_rho1(
            M1, beta, gamma
        )
        return T2_T1

    @staticmethod
    def oblique_rho2_rho1(M1: float, beta: float, gamma: float = 1.4):
        """Returns the density ratio rho2/rho1 across an oblique shock.

        Parameters
        -----------
        M1 : float
            The pre-shock Mach number.

        beta : float
            The shock angle specified in radians.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        T2/T1 : float
            The temperature ratio across the shock.

        References
        ----------
        Peter Jacobs
        """
        M1n = M1 * abs(np.sin(beta))
        rho2_rho1 = (gamma + 1) * M1n**2 / (2 + (gamma - 1) * M1n**2)
        return rho2_rho1

    @staticmethod
    def beta_max(M: float, gamma: float = 1.4):
        """Returns the maximum shock angle for a given
        Mach number.
        """
        beta_max = np.arcsin(
            np.sqrt(
                (1 / (gamma * M**2))
                * (
                    (gamma + 1) * M**2 / 4
                    - 1
                    + np.sqrt(
                        (gamma + 1)
                        * ((gamma + 1) * M**4 / 16 + (gamma - 1) * M**2 / 2 + 1)
                    )
                )
            )
        )
        return beta_max
