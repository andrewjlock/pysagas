import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import bisect
from typing import Optional, Tuple
from pysagas.flow import FlowState
from pysagas.geometry import Vector
from scipy.optimize import root
from pysagas.utilities import add_sens_data
from pysagas.cfd.solver import FlowSolver, FlowResults, SensitivityResults
import time


class OPMVec(FlowSolver):
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

    method = "Oblique/Prandtl-Meyer combination"
    PM_ANGLE_THRESHOLD = -20  # degrees

    def solve(
        self,
        freestream: Optional[FlowState] = None,
        mach: Optional[float] = None,
        aoa: Optional[float] = None,
        cog: Vector = Vector(0, 0, 0),
    ) -> FlowResults:
        already_run = super().solve(freestream=freestream, mach=mach, aoa=aoa)
        if already_run:
            # Already have a result
            if self.verbosity > 1:
                print("Attention: this flowstate has already been solved.")
            result = self.flow_result

        else:
            # Get flow
            flow = self._last_solve_freestream

            # Construct progress bar
            if self.verbosity > 0:
                print()
                desc = f"Running OPM solver at AoA = {flow.aoa:.2f} and Mach = {flow.M:.2f}"
                pbar = tqdm(
                    total=self.cells.num,
                    desc=desc,
                    position=0,
                    leave=True,
                )

            # Iterate over all cells
            net_force = Vector(0, 0, 0)
            net_moment = Vector(0, 0, 0)
            bad = 0
            total = 0

            M2 = np.full(self.cells.num, float(flow.M))
            p2 = np.full(self.cells.num, 0.0)
            T2 = np.full(self.cells.num, float(flow.T))
            method = np.full(self.cells.num, -1)

            theta = np.pi / 2 - np.arccos(
                np.dot(flow.direction.vec, self.cells.n)
                / (
                    np.linalg.norm(self.cells.n, axis=0)
                    * np.linalg.norm(flow.direction.vec)
                )
            )
            r = self.cells.c - cog.vec.reshape(3, 1)
            beta_max = OPMVec.beta_max(M=flow.M, gamma=flow.gamma)
            theta_max = OPMVec.theta_from_beta(
                M1=flow.M, beta=beta_max, gamma=flow.gamma
            )

            ll_idx = np.where(theta < np.deg2rad(self.PM_ANGLE_THRESHOLD))
            l_idx = np.where(
                (np.deg2rad(self.PM_ANGLE_THRESHOLD) < theta) & (theta < 0)
            )
            m_idx = np.where(theta == 0)
            h_idx = np.where((beta_max > theta) & (theta > 0))
            hh_idx = np.where(theta > theta_max)

            M2[l_idx], p2[l_idx], T2[l_idx] = self._solve_pm(
                abs(theta[l_idx]), flow.M, flow.P, flow.T, flow.gamma
            )
            M2[m_idx], p2[m_idx], T2[m_idx] = np.full(
                (len(m_idx), 3), (flow.M, flow.P, flow.T)
            ).T
            M2[h_idx], p2[h_idx], T2[h_idx] = self._solve_oblique(
                abs(theta[h_idx]), flow.M, flow.P, flow.T, flow.gamma
            )
            M2[hh_idx], p2[hh_idx], T2[hh_idx] = np.full(
                (len(hh_idx), 3), self._solve_normal(flow.M, flow.P, flow.T, flow.gamma)
            ).T

            method[ll_idx] = -1
            method[l_idx] = 1
            method[m_idx] = 0
            method[h_idx] = 3
            method[hh_idx] = 2

            self.cells.set_attr("pressure", p2)
            self.cells.set_attr("Mach", M2)
            self.cells.set_attr("temperature", T2)
            self.cells.set_attr("method", method)

            F = self.cells.n * p2 * self.cells.A
            net_force = np.sum(F, axis=1)
            net_moment = np.cross(r.T, F.T).T

            bad = len(ll_idx)

            if self.verbosity > 0:
                pbar.close()

                if bad / self.cells.num > 0.25:
                    print(
                        f"WARNING: {100*bad/self.cells.num:.2f}% of cells were not "
                        "solved due to PM threshold."
                    )

            # Construct results
            result = FlowResults(
                freestream=flow, net_force=net_force, net_moment=net_moment
            )

            # Save
            self.flow_result = result

        # Print result
        if self.verbosity > 1:
            print(result)

        return result

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
        max_step = 100
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
        """Solves for the Mach number, pressure and temperature after a Prandtl-Meyer
        expansion fan.

        Parameters
        ----------
        theta : float
            The deflection angle, specified in radians.

        M1 : float
            The pre-expansion Mach number.

        p1 : float, optional
            The pre-expansion pressure (Pa). The default is 1.0.

        T1 : float, optional
            The pre-expansion temperature (K). The default is 1.0.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        Returns
        --------
        M2 : float
            The post-expansion Mach number.

        p2 : float
            The post-expansion pressure (Pa).

        T2 : float
            The post-expansion temperature (K).
        """
        # Solve for M2
        v_M1 = OPMVec.pm(M=M1, gamma=gamma)
        v_M2 = theta + v_M1
        M2 = OPMVec.inv_pm(v_M2)

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

        M2 = ((1 + M1 * (gamma - 1) / 2) / (gamma * M1**2 - (gamma - 1) / 2)) ** 0.5
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
        b1 = np.arcsin(1.0 / M1) * 0.2
        b2 = np.arcsin(1.0 / M1) * 5

        # Check f1
        f1 = func(b1)
        if np.max(np.abs(f1)) < tolerance:
            return sign_beta * b1

        # Check f2
        f2 = func(b2)
        if np.max(np.abs(f2)) < tolerance:
            return sign_beta * b2

        # Instead of secand method solve again with bisection method
        max_step = 100
        x_0 = np.full(len(theta),b1)
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
        

        # Finally, solve with secant method
        # beta = sign_beta * secant(func, b1, b2, tol=tolerance)
        # root_result = root(f=func, x0=b1, x1=b2, xtol=tolerance, method="secant")
        # if root_result.converged:
        # else:
        #     raise Exception(
        #         f"Cannot converge beta (theta={np.rad2deg(theta):.2f} degrees)."
        #     )

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
        rho2_rho1 = (gamma + 1) * M1n**2 / 2 + (gamma - 1) * M1n**2
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

    def solve_sens(
        self,
        sensitivity_filepath: Optional[str] = None,
        parameters: Optional[list[str]] = None,
        cells_have_sens_data: Optional[bool] = False,
        freestream: Optional[FlowState] = None,
        mach: Optional[float] = None,
        aoa: Optional[float] = None,
        cog: Vector = Vector(0, 0, 0),
        perturbation: float = 1e-3,
    ) -> SensitivityResults:
        # TODO - add to cell attributes for visualisation
        # TODO - return Mach and Temp sensitivities

        already_run = super().solve_sens(freestream=freestream, mach=mach, aoa=aoa)
        if already_run:
            # Already have a result
            if self.verbosity > 1:
                print("Attention: this flowstate sensitivity has already been solved.")
            result = self.flow_sensitivity

        else:
            # Get flow
            flow = self._last_sens_freestream

            # Check that nominal flow condition has been solved already
            if self._last_solve_freestream:
                # Solver has run before
                if self._last_solve_freestream != flow:
                    # Run flow solver first at the nominal flow conditions
                    if self.verbosity > 0:
                        print("Running flow solver to prepare sensitivities.")
                    self.solve(flow)
            else:
                # Solver has not run yet at all, run it now
                self.solve(flow)

            if cells_have_sens_data and not parameters:
                # Need to extract parameters
                _, parameters = self._load_sens_data(sensitivity_filepath)
            elif not cells_have_sens_data:
                # Load sensitivity data and parameters
                sensdata, parameters = self._load_sens_data(sensitivity_filepath)

                # Add sensitivity data to cells
                add_sens_data(cells=self.cells, data=sensdata, verbosity=self.verbosity)

            # Construct progress bar
            if self.verbosity > 0:
                print()
                desc = "Solving cell flow sensitivities."
                pbar = tqdm(
                    total=len(self.cells),
                    desc=desc,
                    position=0,
                    leave=True,
                )

            # Iterate over cells
            dFdp = 0
            dMdp = 0
            for cell in self.cells:
                # Initialisation
                all_directions = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
                sensitivities = np.empty(shape=(cell.dndp.shape[1], 3))
                moment_sensitivities = np.empty(shape=(cell.dndp.shape[1], 3))

                # Calculate moment dependencies
                r = cell.c - cog
                F = cell.flowstate.P * cell.A * cell.n.vec

                # Calculate orientation of cell to flow
                theta = np.pi / 2 - np.arccos(
                    np.dot(flow.direction.vec, cell.n.vec)
                    / (cell.n.norm * flow.direction.norm)
                )

                # Solve flow for this cell
                if theta < 0:
                    # Rearward facing cell
                    if theta < np.deg2rad(self.PM_ANGLE_THRESHOLD):
                        _, dP_dtheta, _ = (0.0, 0.0, 0.0)

                    else:
                        # Use Prandtl-Meyer expansion theory
                        _, dP_dtheta, _ = self.dp_dtheta_pm(
                            abs(theta),
                            flow.M,
                            cell.flowstate,
                            flow.P,
                            flow.T,
                            flow.gamma,
                            perturbation,
                        )

                elif theta > 0:
                    # Use shock theory
                    beta_max = OPMVec.beta_max(M=flow.M, gamma=flow.gamma)
                    theta_max = OPMVec.theta_from_beta(
                        M1=flow.M, beta=beta_max, gamma=flow.gamma
                    )
                    if theta > theta_max:
                        # Detached shock
                        _, dP_dtheta, _ = (0.0, 0.0, 0.0)

                    else:
                        # Use oblique shock theory
                        _, dP_dtheta, _ = self.dp_dtheta_obl(
                            theta,
                            flow.M,
                            cell.flowstate,
                            flow.P,
                            flow.T,
                            flow.gamma,
                            perturbation,
                        )

                else:
                    # Cell is parallel to flow, assume no change
                    _, dP_dtheta, _ = (0.0, 0.0, 0.0)

                # For each parameter
                for p_i in range(cell.dndp.shape[1]):
                    # Calculate gradient of pressure with respect to parameter
                    dtheta_dp = (
                        -np.dot(cell.dndp[:, p_i], flow.direction.vec)
                        / (1 - np.dot(cell.n.unit.vec, flow.direction.unit.vec) ** 2)
                        ** 0.5
                    )

                    if np.isnan(dtheta_dp):
                        # Normal vector parallel to flow
                        dtheta_dp = 0

                    dPdp = dP_dtheta * dtheta_dp

                    # Calculate pressure sensitivity for each direction
                    for i, direction in enumerate(all_directions):
                        dF = (
                            dPdp * cell.A * np.dot(cell.n.vec, direction.vec)
                            + cell.flowstate.P
                            * cell.dAdp[p_i]
                            * np.dot(cell.n.vec, direction.vec)
                            + cell.flowstate.P
                            * cell.A
                            * np.dot(-cell.dndp[:, p_i], direction.vec)
                        )
                        sensitivities[p_i, i] = dF

                    # Now evaluate moment sensitivities
                    moment_sensitivities[p_i, :] = np.cross(
                        r.vec, sensitivities[p_i, :]
                    ) + np.cross(cell.dcdp[:, p_i], F)

                # Append to cell
                cell.sensitivities = sensitivities
                cell.moment_sensitivities = moment_sensitivities

                # Update
                dFdp += sensitivities
                dMdp += moment_sensitivities

                # Update progress bar
                if self.verbosity > 0:
                    pbar.update(1)

            if self.verbosity > 0:
                pbar.close()
                print("Done.")

            # Construct dataframes to return
            df_f = pd.DataFrame(
                dFdp, columns=["dFx/dp", "dFy/dp", "dFz/dp"], index=parameters
            )
            df_m = pd.DataFrame(
                dMdp, columns=["dMx/dp", "dMy/dp", "dMz/dp"], index=parameters
            )

            # Construct results
            result = SensitivityResults(
                f_sens=df_f,
                m_sens=df_m,
                freestream=flow,
            )

            # Save
            self.flow_sensitivity = result

        if self.verbosity > 0:
            print(result)

        return result

    @staticmethod
    def dp_dtheta_obl(
        theta: float,
        M1: float,
        nominal_flowstate: FlowState,
        p1: float = 1.0,
        T1: float = 1.0,
        gamma: float = 1.4,
        perturbation: float = 1e-3,
    ):
        """Solves for the sensitivities at the given point using oblique shock
        theory.

        Parameters
        ----------
        theta : float
            The deflection angle, specified in radians.

        M1 : float
            The pre-expansion Mach number.

        nominal_flowstate : FlowState
            The nominal flowstate at this point.

        p1 : float, optional
            The pre-expansion pressure (Pa). The default is 1.0.

        T1 : float, optional
            The pre-expansion temperature (K). The default is 1.0.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        perturbation : float, optional
            The perturbation amount. Perturbations are calculated
            according to a multiple of (1 +/- perturbation). The
            default is 1e-3.

        Returns
        --------
        dM_dtheta : float
            The sensitivity of the post-expansion Mach number with respect
            to the deflection angle theta.

        dP_dtheta : float
            The sensitivity of the post-expansion pressure (Pa) with respect
            to the deflection angle theta.

        dT_dtheta : float
            The sensitivity of the post-expansion temperature (K) with respect
            to the deflection angle theta.
        """
        # Create function handle
        func = lambda theta: OPMVec._solve_oblique(theta, M1, p1, T1, gamma)

        # Calculate derivitive by finite differencing
        g = OPMVec._findiff(func, theta, nominal_flowstate, perturbation)

        return g

    @staticmethod
    def dp_dtheta_pm(
        theta: float,
        M1: float,
        nominal_flowstate: FlowState,
        p1: float = 1.0,
        T1: float = 1.0,
        gamma: float = 1.4,
        perturbation: float = 1e-3,
    ):
        """Solves for the sensitivities at the given point using Prandtl-Meyer
        theory.

        Parameters
        ----------
        theta : float
            The deflection angle, specified in radians.

        M1 : float
            The pre-expansion Mach number.

        nominal_flowstate : FlowState
            The nominal flowstate at this point.

        p1 : float, optional
            The pre-expansion pressure (Pa). The default is 1.0.

        T1 : float, optional
            The pre-expansion temperature (K). The default is 1.0.

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.

        perturbation : float, optional
            The perturbation amount. Perturbations are calculated
            according to a multiple of (1 +/- perturbation). The
            default is 1e-3.

        Returns
        --------
        dM_dtheta : float
            The sensitivity of the post-expansion Mach number with respect
            to the deflection angle theta.

        dP_dtheta : float
            The sensitivity of the post-expansion pressure (Pa) with respect
            to the deflection angle theta.

        dT_dtheta : float
            The sensitivity of the post-expansion temperature (K) with respect
            to the deflection angle theta.
        """
        # Create function handle
        func = lambda theta: OPMVec._solve_pm(theta, M1, p1, T1, gamma)

        # Calculate derivitive by finite differencing
        g = OPMVec._findiff(func, theta, nominal_flowstate, perturbation)

        return g

    @staticmethod
    def _findiff(
        func: callable,
        point: any,
        nominal_flowstate: FlowState,
        perturbation: float = 1e-3,
    ):
        # Generate flow value at points +/- perturbed from nominal point
        xvals = [point * (1 - perturbation), point, point * (1 + perturbation)]
        vals = [
            func(p) for p in [point * (1 - perturbation), point * (1 + perturbation)]
        ]
        Mvals = [v[0] for v in vals]
        pvals = [v[1] for v in vals]
        Tvals = [v[2] for v in vals]

        # Inject nominal result
        Mvals = [Mvals[0], nominal_flowstate.M, Mvals[1]]
        pvals = [pvals[0], nominal_flowstate.P, pvals[1]]
        Tvals = [Tvals[0], nominal_flowstate.T, Tvals[1]]

        dM = np.gradient(Mvals, xvals)
        dp = np.gradient(pvals, xvals)
        dT = np.gradient(Tvals, xvals)

        return dM[1], dp[1], dT[1]

    def save(self, name: str):
        # Initialise attributes dictionary
        attributes = {
            "pressure": [],
            "Mach": [],
            "temperature": [],
            "method": [],
        }

        # Call super method
        super().save(name, attributes)

    @staticmethod
    def _load_sens_data(sensitivity_filepath: str) -> Tuple[pd.DataFrame, list[set]]:
        """Extracts design parameters from the sensitivity data."""
        sensdata = pd.read_csv(sensitivity_filepath)

        parameters = set()
        for e in sensdata.columns:
            e: str
            if e.startswith("dxd") or e.startswith("dyd") or e.startswith("dzd"):
                # Sensitivity coluns
                parameters.add(e[3:])

        return sensdata, list(parameters)
