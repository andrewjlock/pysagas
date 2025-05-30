import numpy as np
from typing import Union
from pysagas.sensitivity.models_vec import van_dyke_sensitivity, freestream_isentropic_sensitivity
from pysagas.flow import FlowStateVec, FlowState, InFlowStateVec
from utilities import PatchTag

def sensitivity_calculator_vec(
    cells,
    freestream: Union[FlowState, InFlowStateVec],
    flow_state,
    inflow_sens= None,
    cog=np.array([0, 0, 0]),
    cog_sens=None,
    A_ref: float = 1,
    c_ref: float = 1,
):
    if isinstance(freestream, FlowState):
        flow = freestream
        inflow = freestream
    elif isinstance(freestream, InFlowStateVec):
        flow = freestream.freestream
        inflow = freestream
    else:
        raise ValueError("Invalid freestream object.")

    sensitivity_function = van_dyke_sensitivity
    # sensitivity_function = piston_sensitivity

    sensitivities = np.zeros(shape=(cells.dAdp.shape[0], 3, cells.num))
    moment_sensitivities = np.zeros(shape=(cells.dAdp.shape[0], 3, cells.num))
    dPdp_c = np.full(cells.num, 0.0)
    dPdp_e = np.full(cells.num, 0.0)

    # # INLET and OUTLET cells aerodynamics are not computed
    calc_idx = (cells.tag != PatchTag.INLET.value) | (cells.tag != PatchTag.OUTLET.value)
    eng_idx = (cells.tag == PatchTag.NOZZLE.value)



    for p in range(cells.dAdp.shape[0]):
        # dPdp = sensitivity_function(cells, flow_state, p)
        dPdp_c[calc_idx] = sensitivity_function(cells, flow_state, p, calc_idx)
        if any(eng_idx):
            dPdp_e[eng_idx] = freestream_isentropic_sensitivity(cells=cells, flowstate=flow_state, p_i=p, inflow=inflow, inflow_sens=inflow_sens, eng_idx=eng_idx)
        dPdp = dPdp_c + dPdp_e

        dF = (
            dPdp * cells.A * -cells.n
            + flow_state.p * cells.dAdp[p] * -cells.n
            + flow_state.p * cells.A * -cells.dndp[p, :, :]
        )
        sensitivities[p, :, :] = dF

        r = cells.c - cog.reshape(3, 1)
        F = flow_state.p * cells.A * -cells.n

        if cog_sens is None:
            moment_sensitivities[p, :, :] = (
                np.cross(r.T, sensitivities[p, :, :].T).T
                + np.cross(cells.dcdp[p, :, :], F.T).T
            )
        else:
            moment_sensitivities[p, :, :] = (
                np.cross(r.T, sensitivities[p, :, :].T).T
                + np.cross(cells.dcdp[p, :, :] - cog_sens[p], F.T).T
            )

        flow_state.p_sens.append(dPdp)

    #TODO - change names to DFdp (for force), and dCFdp for coefficent
    dForcedp = np.sum(sensitivities, axis=-1)
    dMomentdp = np.sum(moment_sensitivities, axis=-1)
    dFdp = dForcedp / (flow.q * A_ref)
    dMdp = dMomentdp / (flow.q * A_ref * c_ref)

    result = {
        "dFdp": dFdp,
        "dMdp": dMdp,
        "dForcedp": dForcedp,
        "dMomentdp": dMomentdp,
    }
    return result
