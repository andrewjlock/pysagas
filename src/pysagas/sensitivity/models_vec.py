import numpy as np
from pysagas.geometry import CellArray


def piston_sensitivity(cell: CellArray, flowstate, p_i: int, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    local piston theory.

    Parameters
    ----------
    cell : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.
    """

    ss_idx = np.where(flowstate.M > 1)  # Cells with supersonic flow
    dPdp = np.zeros((cell.num))

    dPdp[ss_idx] = (
        flowstate.rho[ss_idx]
        * flowstate.a[ss_idx]
        * np.einsum(
            "i...,i...", flowstate.vec[:, ss_idx], -cell.dndp[:, p_i, ss_idx]
        ).reshape(-1)
    )
    return dPdp


def van_dyke_sensitivity(
    cell: CellArray,
    flowstate,
    p_i,
    **kwargs,
):
    """
    Calculates the pressure-parameter sensitivity using
    Van Dyke second-order theory.

     Parameters
    ----------
    cell : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.
    """

    ss_idx = np.where(flowstate.M > 1)  # Cells with supersonic flow
    dPdp = np.zeros(cell.num)

    piston = piston_sensitivity(cell=cell, flowstate=flowstate, p_i=p_i)
    dPdp[ss_idx] = (
        piston[ss_idx] * flowstate.M[ss_idx] / (flowstate.M[ss_idx] ** 2 - 1) ** 0.5
    )
    return dPdp


# def isentropic_sensitivity(cell: CellArray, p_i: int, **kwargs):
#     """Calculates the pressure-parameter sensitivity using
#     the isentropic flow relation directly."""
#     # TODO: Probably need to fix tensor multiplaction here...
#     gamma = 1.4
#     power = (gamma + 1) / (gamma - 1)
#     dPdW = (cell.pressure * gamma / cell.a) * (
#         1 + cell.v_mag * (gamma - 1) / (2 * cell.a)
#     ) ** power
#     dWdn = -cell.vec
#     dndp = cell.dndp[:, :, p_i]
#     dPdp = dPdW * np.dot(dWdn, dndp)
#     return dPdp
