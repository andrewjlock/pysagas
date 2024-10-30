import numpy as np
import pandas as pd
from pysagas.geometry import CellArray
from pysagas.sensitivity.models_vec import van_dyke_sensitivity
import time


def sensitivity_calculator_vec(cells, flow_state, cog = np.array([0,0,0])):

    t1 = time.time()
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

        r = cells.c - cog.reshape(3,1)
        F = flow_state.p * cells.A * cells.n
        moment_sensitivities[p, :, :] = np.cross(
            r.T, sensitivities[p, :, :].T
        ).T + np.cross(cells.dcdp[:, p].T, F.T).T

    cells.sensitivities = sensitivities
    cells.moment_sensitivities = moment_sensitivities
    t2 = time.time()
    print(t2-t1)

    breakpoint()
