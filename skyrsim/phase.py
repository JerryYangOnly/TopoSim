import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

import multiprocessing as mp
import typing

from .model import *
from .bulk import Simulator

class ModelWrapper:
    def __init__(self, model: type(Model), param_x: str, param_y: str):
        self.model = model
        self.parameters = self.model.defaults
        if param_x not in self.parameters:
            raise ValueError("Requested parameter `" + param_x + "` is not accepted by model `" + self.model.name + "`.")
        if param_y not in self.parameters:
            raise ValueError("Requested parameter `" + param_y + "` is not accepted by model `" + self.model.name + "`.")
        self.param_x = param_x
        self.param_y = param_y
        self.parameters.pop(self.param_x)
        self.parameters.pop(self.param_y)

    def set_value(self, param: str, value) -> None:
        if param == self.param_x or param == self.param_y:
            raise ValueError("Parameter `" + param + "` is an independent variable.")
        if param not in self.parameters:
            raise ValueError("Requested parameter `" + param + "` is not accepted by model `" + self.model.name + "`.")
        self.parameters[param] = value

    def __call__(self, x, y) -> Model:
        return self.model({**self.parameters, self.param_x: x, self.param_y: y})


class PhaseDiagram:
    def __init__(self, model: ModelWrapper, xlim: typing.Union[tuple, np.ndarray],
            ylim: typing.Union[tuple, np.ndarray], sim_density: int=101):
        self.model = model

        if isinstance(xlim, tuple):
            if len(xlim) != 3:
                raise ValueError("Expected a tuple of length 3 for `xlim`.")
            self.xlim = np.linspace(xlim[0], xlim[1], xlim[2])
        else:
            self.xlim = xlim

        if isinstance(ylim, tuple):
            if len(ylim) != 3:
                raise ValueError("Expected a tuple of length 3 for `ylim`.")
            self.ylim = np.linspace(ylim[0], ylim[1], ylim[2])
        else:
            self.ylim = ylim

        if isinstance(self.model, ModelWrapper):
            self.xlabel = self.model.param_x
            self.ylabel = self.model.param_y
        else:
            self.xlabel = ""
            self.ylabel = ""

        self.title = ""
        self.sim_density = sim_density
        self.S = None
        self.filled_bands = None

        self.chern = self.skyr = self.z2 = self.skyr_z2 = self.gap = self.spin_gap = False
        self.result = {}

    def set_xlabel(self, label: str) -> None:
        self.xlabel = label

    def set_ylabel(self, label: str) -> None:
        self.ylabel = label

    def set_title(self, label: str) -> None:
        self.title = label

    def set_spin_op(self, S: np.ndarray) -> None:
        self.S = S

    def set_filled_bands(self, filled_bands: int) -> None:
        self.filled_bands = filled_bands

    def _compute(self, x, y):
        sim = Simulator(self.model(x, y), self.sim_density)
        sim.set_spin_op(self.S)
        res = []
        if self.chern:
            res.append(sim.compute_chern(self.filled_bands))
        if self.skyr:
            res.append(sim.compute_skyrmion(self.filled_bands))
        if self.z2:
            res.append(sim.compute_z2(self.filled_bands))
        if self.skyr_z2:
            res.append(sim.compute_skyrmion_z2(self.S, self.filled_bands))
        if self.gap:
            res.append(sim.direct_band_gap(self.filled_bands))
        if self.spin_gap:
            res.append(sim.minimum_spin_gap(self.filled_bands))
        return tuple(res)

    def generate(self, invar: list = [], max_cpu: int=0) -> None:
        if invar == []:
            invar = ["chern", "skyr", "z2", "skyr_z2", "gap", "spin_gap"]
        if max_cpu == 0:
            max_cpu = mp.cpu_count() // 2

        invar = [s for s in invar if s in ["chern", "skyr", "z2", "skyr_z2", "gap", "spin_gap"]]
        self.chern = "chern" in invar
        self.skyr = "skyr" in invar
        self.z2 = "z2" in invar
        self.skyr_z2 = "skyr_z2" in invar
        self.gap = "gap" in invar
        self.spin_gap = "spin_gap" in invar
        
        if (self.skyr or self.skyr_z2 or self.spin_gap) and self.S is None:
            raise ValueError("Spin related quantities cannot be evaluated with a spin operator set.")

        X, Y = np.meshgrid(self.xlim, self.ylim)
        with mp.Pool(max_cpu) as pool:
            result = pool.starmap(self._compute, zip(X.flatten(), Y.flatten()))
        result = np.array(result).T
        for i in range(len(invar)):
            self.result[invar[i]] = result[i].reshape(self.xlim.shape[0], self.ylim.shape[0])

    def save(self, filename: str) -> None:
        np.savez(filename, **self.result)

    def load(self, filename: str) -> None:
        self.result = np.load(filename)

    def plot(self, invar: list = [], label: typing.Union[bool, dict] = True, title: typing.Union[bool, dict] = False) -> None:
        if invar == []:
            invar = ["chern", "skyr", "z2", "skyr_z2", "gap", "spin_gap"]

        for key in self.result:
            if key not in invar:
                continue
            fig = plt.figure()
            ax = fig.gca()
            dx = (self.xlim[-1] - self.xlim[0]) / (self.xlim.shape[0] - 1)
            dy = (self.ylim[-1] - self.ylim[0]) / (self.ylim.shape[0] - 1)
            extent = [self.xlim[0] - dx/2, self.xlim[-1] + dx/2, self.ylim[0] - dy/2, self.ylim[-1] + dy/2]
            aspect = self.xlim.shape[0] / self.ylim.shape[0]
            pos = ax.imshow(self.result[key], aspect=aspect, extent=extent, origin="lower")
            cb = fig.colorbar(pos, ax=ax)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

            if isinstance(label, dict):
                cb.set_label(label[key])
            elif label is True:
                cb.set_label({"chern": "$\\mathcal{C}$", "skyr": "$\\mathcal{Q}$", "z2": "$\\nu$",
                    "skyr_z2": "$\\nu_Q$", "gap": "$\\Delta E$", "spin_gap": "\\Delta |S|"}[key])

            if isinstance(title, dict):
                ax.set_title(title[key])
            elif title is True:
                ax.set_title({"chern": "Chern number$\\mathcal{C}$",
                    "skyr": "Skyrmion number $\\mathcal{Q}$",
                    "z2": "Z2 invariant $\\nu$",
                    "skyr_z2": "Skyrmion Z2 invariant $\\nu_Q$",
                    "gap": "$Direct band gap \\Delta E$",
                    "spin_gap": "Minimum spin \\Delta |S|"}[key])
            
            fig.savefig("_".join([self.model.param_x, self.model.param_y, key]) + ".png", dpi=600)
            plt.close(fig)

    def subplots(self, shape: tuple, invar: list = [], label: typing.Union[bool, dict] = True) -> None:
        if invar == []:
            invar = ["chern", "skyr", "z2", "skyr_z2", "gap", "spin_gap"]

        fig, ax = plt.subplots(*shape)
        count = 0
        
        for key in self.result:
            sax = ax[count // shape[1], count % shape[1]]
            if key not in invar:
                continue
            dx = (self.xlim[-1] - self.xlim[0]) / (self.xlim.shape[0] - 1)
            dy = (self.ylim[-1] - self.ylim[0]) / (self.ylim.shape[0] - 1)
            extent = [self.xlim[0] - dx/2, self.xlim[-1] + dx/2, self.ylim[0] - dy/2, self.ylim[-1] + dy/2]
            aspect = self.xlim.shape[0] / self.ylim.shape[0]
            pos = sax.imshow(self.result[key], aspect=aspect, extent=extent, origin="lower")
            cb = fig.colorbar(pos, ax=sax)
            sax.set_xlabel(self.xlabel)
            sax.set_ylabel(self.ylabel)

            if isinstance(label, dict):
                cb.set_label(label[key])
            elif label is True:
                cb.set_label({"chern": "$\\mathcal{C}$", "skyr": "$\\mathcal{Q}$", "z2": "$\\nu$",
                    "skyr_z2": "$\\nu_Q$", "gap": "$\\Delta E$", "spin_gap": "\\Delta |S|"}[key])

        fig.suptitle(self.title)
        fig.savefig("_".join([self.model.param_x, self.model.param_y]) + ".png", dpi=600)
        plt.close(fig)
