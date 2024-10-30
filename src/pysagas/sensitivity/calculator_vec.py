import numpy as np
import pandas as pd
from pysagas.geometry import CellArray
from pysagas.sensitivity.models_vec import van_dyke_sensitivity


def sensitivity_calculator_vec(
    cells,
    freestream,
    flow_state,
    cog=np.array([0, 0, 0]),
    A_ref: float = 1,
    c_ref: float = 1,
):

    sensitivity_function = van_dyke_sensitivity

    sensitivities = np.zeros(shape=(cells.dAdp.shape[0], 3, cells.num))
    moment_sensitivities = np.zeros(shape=(cells.dAdp.shape[0], 3, cells.num))

    for p in range(cells.dAdp.shape[0]):
        dPdp = van_dyke_sensitivity(cells, flow_state, p)
        dF = (
            dPdp * cells.A * cells.n
            + flow_state.p * cells.dAdp[p] * cells.n
            - flow_state.p * cells.A * cells.dndp[:, p, :]
        )
        sensitivities[p, :, :] = dF

        r = cells.c - cog.reshape(3, 1)
        F = flow_state.p * cells.A * cells.n
        moment_sensitivities[p, :, :] = (
            np.cross(r.T, sensitivities[p, :, :].T).T
            + np.cross(cells.dcdp[:, p].T, F.T).T
        )

    dFdp = np.sum(sensitivities, axis=-1) / (freestream.q * A_ref)
    dMdp = np.sum(moment_sensitivities, axis=-1) / (freestream.q * A_ref * c_ref)
    result = {
        "dFdp": dFdp,
        "dMdp": dMdp,
    }
    return result
