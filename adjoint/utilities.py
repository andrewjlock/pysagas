import numpy as np
from typing import List, Callable
import gdtk.ideal_gas_flow as igf
from adjoint.flow import FlowState
from adjoint.geometry import Vector, Cell


def calculate_pressures(flow: FlowState, theta: float) -> float:
    """Calculates the pressure from a flow state and deflecion angle
    using ideal gas oblique shock theory.

    Parameters
    ----------
    flow : FlowState
        The flow state.
    theta : float
        The deflection angle (radians).

    Returns
    --------
    P2 : float
        The pressure behind the oblique shock.
    """
    beta = igf.beta_obl(M1=flow.M, theta=abs(theta), g=flow.gamma, tol=1.0e-6)
    P2_P1 = igf.p2_p1_obl(flow.M, beta, g=flow.gamma)
    P2 = P2_P1 * flow.P
    return P2


def calculate_force_vector(P: float, n: np.array, A: float) -> np.array:
    """Calculates the force vector components, acting on a
    surface defined by its area and normal vector.

    Parameters
    ----------
    P : float
        The pressure.
    n : np.array
        The normal vector.
    A : float
        The reference area (m^2).

    Returns
    --------
    forces : np.array
        The force components.
    """
    F_x = A * P * np.dot(n, np.array([-1, 0, 0]))
    F_y = A * P * np.dot(n, np.array([0, -1, 0]))
    F_z = A * P * np.dot(n, np.array([0, 0, -1]))

    return [F_x, F_y, F_z]


def cell_dfdp(cell: Cell, dPdp_method: Callable, **kwargs) -> np.array:
    """Calculates all direction force sensitivities.

    Parameters
    ----------
    cell : Cell
        The cell.

    Returns
    --------
    sensitivities : np.array
        An array of shape n x 3, for a 3-dimensional cell with
        n parameters.

    See Also
    --------
    all_dfdp : a wrapper to calculate force sensitivities for many cells
    """
    all_directions = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

    sensitivities = np.empty(shape=(cell.dndp.shape[1], 3))
    for p_i in range(cell.dndp.shape[1]):
        # Calculate pressure sensitivity
        dPdp = dPdp_method(cell=cell, p_i=p_i, **kwargs)

        # Evaluate force sensitivity for each direction
        for i, direction in enumerate(all_directions):
            dir_sens = (
                dPdp * cell.A * np.dot(cell.n.vec, direction.vec)
                + cell.flowstate.P * cell.dAdp[p_i] * np.dot(cell.n.vec, direction.vec)
                + cell.flowstate.P * cell.A * np.dot(-cell.dndp[:, p_i], direction.vec)
            )
            sensitivities[p_i, i] = dir_sens

    return sensitivities


def panel_dPdp(cell: Cell, p_i, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    the Panel method approximation."""
    dPdp = (
        cell.flowstate.rho
        * cell.flowstate.a
        * np.dot(cell.flowstate.vec, -cell.dndp[:, p_i])
    )
    return dPdp


def isentropic_dPdp(cell: Cell, p_i: int, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    the isentropic flow relation directly."""
    # TODO - direct Pressure method. Unsure what to use for P_inf
    gamma = cell.flowstate.gamma
    power = (gamma + 1) / (gamma - 1)
    dPdW = (
        kwargs["P_inf"]
        * gamma
        * (2 * cell.flowstate.a + cell.flowstate.v * (gamma - 1)) ** power
    ) / (2**power * cell.flowstate.a ** (2 * gamma / (gamma - 1)))
    dWdn = cell.flowstate.v
    dndp = cell.dndp[:, p_i]
    dPdp = np.linalg.multi_dot(arrays=[dPdW, dWdn, dndp])
    return dPdp


def all_dfdp(cells: List[Cell], dPdp_method: Callable = panel_dPdp) -> np.array:
    """Calcualtes the force sensitivities for a list of Cells.

    Parameters
    ----------
    cells : list[Cell]
        The cells to be analysed.

    Returns
    --------
    dFdp : np.array
        The force sensitivity matrix with respect to the parameters.

    See Also
    --------
    cell_dfdp : the force sensitivity per cell
    """
    dFdp = 0
    for cell in cells:
        # Calculate force sensitivity
        dFdp += cell_dfdp(cell=cell, dPdp_method=dPdp_method)

    return dFdp
