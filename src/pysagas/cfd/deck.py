import copy
import pandas as pd
from typing import Optional, Union
from abc import ABC, abstractmethod


class AeroDeck:
    def __init__(self, inputs):
        self.inputs = inputs
        columns = inputs + ["CF1", "CF2", "CF3", "CM1", "CM2", "CM3"]
        self.df = pd.DataFrame(columns=columns)

    def insert(self, inputs, result_dict):
        row = inputs + list(result_dict["CF"]) + list(result_dict["CM"])
        self.df.loc[len(self.df.index)] = row

    def to_csv(self):
        self.df.to_csv("aerodeck.csv")


class SensDeck:
    def __init__(self, inputs, parameters):
        self.inputs = inputs
        self.parameters = parameters
        self.n = len(self.parameters)
        data_headers = ["CF1", "CF2", "CF3", "CM1", "CM2", "CM3"]
        self.dfs = []
        for param in parameters:
            headers = []
            for data in data_headers:
                headers.append("d" + data + "/d" + param)
            self.dfs.append(pd.DataFrame(columns=inputs + headers))

    def insert(self, inputs, result_dict):
        for i in range(self.n):
            row = inputs + list(result_dict["dFdp"][i]) + list(result_dict["dMdp"][i])
            self.dfs[i].loc[len(self.dfs[i].index)] = row

    def to_csv(self):
        for i, p in enumerate(self.parameters):
            # Change headers back to only derivative numerator for compatability
            # with existing codes (may change in future)

            # headers = self.dfs[i].columns.values.tolist()
            # headers_ = [h.split("_")[0] for h in headers]
            # self.dfs[i].rename(
            #     columns={h: h_ for h, h_ in zip(headers, headers_)}, inplace=True
            # )

            self.dfs[i].to_csv(p+"_sensdeck.csv")
