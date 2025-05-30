import pandas as pd


class AeroDeck:
    def __init__(self, inputs, add_coeffs=True, add_forces=False):
        self.inputs = inputs
        self.add_coeffs = add_coeffs
        self.add_forces = add_forces

        if not (self.add_forces or self.add_coeffs):
            raise ValueError("At least one of add_forces or add_coeffs must be True")

        columns = []
        if self.add_coeffs:
            columns += ["CF1", "CF2", "CF3", "CM1", "CM2", "CM3"]
        if self.add_forces:
            columns += ["F1", "F2", "F#", "M1", "M2", "M3"]

        self.df = pd.DataFrame(columns=inputs + columns)

    def insert(self, inputs, result_dict):

        row = inputs[:]
        if self.add_coeffs:
            row += list(result_dict["CF"]) + list(result_dict["CM"])
        if self.add_forces:
            row += list(result_dict["F"]) + list(result_dict["M"])
        self.df.loc[len(self.df.index)] = row

    def to_csv(self):
        self.df.to_csv("aerodeck.csv")




class SensDeck:
    def __init__(self, inputs, parameters, add_coeffs=True, add_forces=False):
        self.inputs = inputs
        self.add_coeffs = add_coeffs
        self.add_forces = add_forces
        self.parameters = parameters
        self.n = len(self.parameters)

        if not (self.add_forces or self.add_coeffs):
            raise ValueError("At least one of add_forces or add_coeffs must be True")

        data_headers = []
        if self.add_coeffs:
            data_headers += ["CF1", "CF2", "CF3", "CM1", "CM2", "CM3"]
        if self.add_forces:
            data_headers += ["F1", "F2", "F3", "M1", "M2", "M3"]

        self.dfs = []
        for param in parameters:
            headers = []
            for data in data_headers:
                headers.append("d" + data + "/d" + param)
            self.dfs.append(pd.DataFrame(columns=inputs + headers))

    def insert(self, inputs, result_dict):
        for i in range(self.n):
            row = inputs[:]
            if self.add_coeffs:
                row += list(result_dict["dFdp"][i]) + list(result_dict["dMdp"][i])
            if self.add_forces:
                row += list(result_dict["dForcedp"][i]) + list(result_dict["dMomentdp"][i])
            self.dfs[i].loc[len(self.dfs[i].index)] = row

    def to_csv(self):
        for i, p in enumerate(self.parameters):
            self.dfs[i].to_csv(p + "_sensdeck.csv")
