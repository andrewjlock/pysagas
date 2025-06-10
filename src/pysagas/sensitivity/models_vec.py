import numpy as np

from pysagas.geometry import CellArray
from pysagas.flow import FlowStateVec, FlowState, InFlowStateVec


def piston_sensitivity(cells: CellArray, flowstate, p_i: int, calc_idx=None, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    local piston theory.

    Parameters
    ----------
    cells : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.

    calc_idx : list, optional
        Cell id to be calculated, by default None. If None, all cells will be calculated.
    """

    if calc_idx is None:
        calc_idx = np.ones(cells.num, dtype=bool)

    ss_idx = np.where((flowstate.M > 1) & (calc_idx))  # Cells with supersonic flow
    dPdp = np.zeros((cells.num))

    dPdp[ss_idx] = (
        flowstate.rho[ss_idx]
        * flowstate.a[ss_idx]
        * np.einsum(
            "i...,...i", flowstate.vec[:, ss_idx], -cells.dndp[p_i, :, ss_idx]
        ).reshape(-1)
    )
    return dPdp


def van_dyke_sensitivity(
    cells: CellArray,
    flowstate,
    p_i,
    calc_idx=None,
    **kwargs,
):
    """
    Calculates the pressure-parameter sensitivity using
    Van Dyke second-order theory.

     Parameters
    ----------
    cells : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.
    calc_idx : list, optional
        Cell id to be calculated, by default None. If None, all cells will be calculated.
    """

    if calc_idx is None:
        calc_idx = np.ones(cells.num, dtype=bool)

    ss_idx = np.where((flowstate.M > 1) & (calc_idx))  # Cells with supersonic flow
    dPdp = np.zeros(cells.num)

    piston = piston_sensitivity(
        cells=cells, flowstate=flowstate, p_i=p_i, calc_idx=calc_idx
    )
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


def freestream_isentropic_sensitivity(
    cells: CellArray,
    flowstate: FlowStateVec,
    p_i: int,
    inflow: InFlowStateVec,
    inflow_sens,
    eng_idx,
    **kwargs,
):
    """Calculates the pressure-parameter sensitivity, including
    the sensitivity to the incoming flow state (for use on nozzle cells
    where the engine outflow changes due to parameter change)

    Parameters
        ----------
        cells : CellArray
            The cell object.

        flowstate : FlowStateVec
            The nominal FlowState correspondnig to each cell.

        p_i : int
            Index of the design variable (in inflow_sens) to differentiate with respect to

        inflow : InFlowStateVec
            "Free stream" of the incoming flow of each cell (should be the engine combustor outflow for nozzle cells)

        inflow_sens :
            HyperPro engine outflow sensitivities

        Returns
        --------
        dPdp : np.array
            The pressure sensitivity matrix with respect to the parameter.
    """

    gamma1 = inflow.gamma[eng_idx]
    gamma2 = flowstate.gamma
    M1 = inflow.M[eng_idx]
    M2 = flowstate.M[eng_idx]
    P1 = inflow.P[eng_idx]
    P2 = flowstate.p[eng_idx]

    beta1 = np.sqrt(M1**2 - 1)
    beta2 = np.sqrt(M2**2 - 1)
    fun1 = 1 + (gamma1 - 1) / 2 * M1**2
    fun2 = 1 + (gamma2 - 1) / 2 * M2**2

    # Calculate sens to inflow Mach number
    dM2_dM1 = (M2 / M1) * (beta1 / beta2) * (fun2 / fun1)
    t1 = M1 * fun1 ** (1 / (gamma1 - 1)) * fun2 ** ((-gamma1) / (gamma1 - 1))
    t2 = (
        M2
        * fun1 ** (gamma1 / (gamma1 - 1))
        * fun2 ** ((1 - 2 * gamma1) / (gamma1 - 1))
        * dM2_dM1
    )
    dP2_dM1 = P1 * gamma1 * (t1 - t2)

    # Calculate sens to inflow pressure
    dP2_dP1 = (fun1 / fun2) ** (gamma2 / (gamma2 - 1))

    # Calculate sens to inflow aoa
    dP2_daoa = M2**2 / beta2 * gamma2 * P2

    # Calculate sens to inflow gamma
    gp1 = gamma1 + 1
    gm1 = gamma1 - 1
    fg = gp1 / gm1
    fM = beta1**2 / fg
    num1 = np.sqrt(fg) * (beta1 / gp1) ** 2
    den1 = np.sqrt(fM) * (fM + 1)
    num2 = 1 / gm1 * np.arctan(np.sqrt(fM))
    den2 = np.sqrt(gm1 * gp1)
    dnu_dg1 = num1 / den1 - num2 / den2

    q = (fun1 / fun2) ** (gamma1 / gm1)
    r1 = gamma1 * (M1**2 - (M2**2 * fun1) / fun2)
    r2 = 2 * gm1 * fun1
    s1 = 1 / gm1 - gamma1 / (gm1**2)
    s2 = np.log(fun1 / fun2)
    df_dg = q * (r1 / r2 + s1 * s2)

    dP2_dg1 = P1 * df_dg - dP2_daoa * dnu_dg1

    # sum contributions
    dPdp = (
        dP2_dM1 * inflow_sens.loc["M"][p_i]
        + dP2_dP1 * inflow_sens.loc["P"][p_i]
        + dP2_daoa * inflow_sens.loc["flow_angle"][p_i]
        + dP2_dg1 * inflow_sens.loc["gamma"][p_i]
    )

    return dPdp
