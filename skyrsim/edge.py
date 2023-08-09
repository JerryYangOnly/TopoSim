import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from .common import *
from .model import *
from .bulk import Simulator

class EdgeSimulator(Simulator):
    def __init__(self, model: Model, points: int):
        super().__init__(model, points)

        self.N = np.zeros(self.model.dim)
        self.PBC = np.array([True] * self.model.dim)
        self.open_dim = []
        self.eff_dim = self.model.dim
        self.eff_bands = self.model.bands

    def open(self, N):
        self.N = np.array(N)
        self.open_dim = np.arange(self.model.dim)[self.N > 0]
        self.PBC[self.open_dim] = False
        
        self.eff_dim = self.model.dim - len(self.open_dim)
        self.eff_bands = self.model.bands * np.prod(self.N[self.open_dim])
        self.band = np.zeros((*([self.mesh_points] * self.eff_dim), self.eff_bands))
        self.states = np.zeros((*([self.mesh_points] * self.eff_dim), self.eff_bands, self.eff_bands), dtype=np.complex64)
         
        self.evaluated = False

    def set_PBC(self, PBC):
        self.PBC = np.array(PBC)

    def populate_mesh(self):
        if self.evaluated:
            return

        self.band = self.band.reshape(self.mesh_points**self.eff_dim, self.eff_bands)
        self.states = self.states.reshape(self.mesh_points**self.eff_dim, self.eff_bands, self.eff_bands)
        for i in range(self.mesh_points**self.eff_dim):
            idx = [0] * self.model.dim
            k = i
            for j in range(self.model.dim - 1, -1, -1):
                if j not in self.open_dim:
                    idx[j] = k % self.mesh_points
                    k //= self.mesh_points
            w, v = scipy.linalg.eigh(self.model.open_hamiltonian(self.N, [self.mesh[j] for j in idx], PBC=self.PBC))
            self.band[i, :] = w
            self.states[i, :, :] = v

        self.band = self.band.reshape((*([self.mesh_points] * self.eff_dim), self.eff_bands))
        self.states = self.states.reshape((*([self.mesh_points] * self.eff_dim), self.eff_bands, self.eff_bands))
        self.evaluated = True

    def plot_band(self, band_hl=(), pi_ticks=True, close_fig=False, return_fig=False, save_fig=""):
        if not self.evaluated:
            self.populate_mesh()
        if save_fig:
            close_fig = True
        if close_fig and return_fig:
            raise ValueError("`close_fig` and `return_fig` cannot both be True")
        if self.eff_dim == 1:
            fig, ax = plt.subplots()
            for i in range(self.eff_bands):
                if i not in band_hl:
                    ax.plot(self.mesh, self.band[:, i], "k-")
                else:
                    ax.plot(self.mesh, self.band[:, i])
            
            dims = [i for i in range(self.model.dim) if i not in self.open_dim]
            dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
            ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
            ax.set_ylabel("$E(\\mathbf{k})$")
            ax.set_title("Band spectrum")
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])

            if save_fig:
                fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            if return_fig:
                return fig

        elif self.eff_dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)

            for i in range(self.eff_bands):
                ax.plot_surface(X, Y, self.band[:, :, i])
            del X, Y

            dims = [i for i in range(self.model.dim) if i not in self.open_dim]
            dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
            ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
            ax.set_ylabel("$k_{" + dim_labels(dims[1]) + "}$")
            ax.set_zlabel("$E(\\mathbf{k})$")
            ax.set_title("Band spectrum")
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
                ax.set_yticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            if save_fig:
                fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            elif return_fig:
                return fig

        else:
            raise ValueError("Band plotting of models in %d-D is not supported" % self.model.dim)

    def in_gap_states(self, n_states=2, fermi=0.0):
        ids = np.argpartition(np.abs(self.band - fermi), n_states, axis=None)
        ids = np.array(np.unravel_index(ids[:n_states], self.band.shape)).transpose()
        return [self.states.__getitem__(*idx[:-1])[:, idx[-1]] for idx in ids]

    def pdf(self, state, sum_internal=True):
        s = state.reshape(tuple(self.N[self.open_dim]) + (self.model.bands,), order='C')
        s = np.abs(s)**2
        # s /= np.sum(s)      # Normalization

        if sum_internal:
            s = np.sum(s, axis=len(self.open_dim))
        return s


    def pdfs(self, states, sum_internal=True):
        return [self.pdf(state, sum_internal=sum_internal) for state in states]
        
    def position_heat_map_band(self, band, pi_ticks=True, close_fig=False, return_fig=False, save_fig="", max_magnitude=1.0, cmap="hot"):
        if self.eff_dim != 1 or len(self.open_dim) != 1:
            raise ValueError("Dimension of the model is not supported. Are boundaries opened correctly?")
        if save_fig:
            close_fig = True
        if close_fig and return_fig:
            raise ValueError("`close_fig` and `return_fig` cannot both be True")
        if not self.evaluated:
            self.populate_mesh()
        
        fig = plt.figure()
        # ax = fig.gca(projection="3d")
        ax = fig.gca()

        pdfs = np.array(self.pdfs(self.states[:, :, band], sum_internal=True))

        im = ax.imshow(pdfs.transpose(), aspect=2 * np.pi / np.prod(self.N[self.open_dim]),
                  extent=(-np.pi, np.pi, 0, np.prod(self.N[self.open_dim])), origin="lower",
                  vmin=0.0, vmax=max_magnitude, cmap=cmap)
        fig.colorbar(im, ax=ax)

        dims = [i for i in range(self.model.dim) if i not in self.open_dim]
        dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
        ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
        ax.set_ylabel("Site")
        ax.set_title("Probability distributions of band %d" % (band + 1))

        if pi_ticks:
            ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
        # ax.set_zlim(0, 1)

        if save_fig:
            fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
        else:
            plt.show()
        if close_fig:
            plt.close(fig)
        elif return_fig:
            return fig
        
    def plot_spin_band(self, band, pi_ticks=True, close_fig=False, return_fig=False, save_fig="", subplots=False):
        if self.eff_dim != 1:
            raise ValueError("Spin plotting of bands is only supported in 1-D")
        if save_fig:
            close_fig = True
        if close_fig and return_fig:
            raise ValueError("`close_fig` and `return_fig` cannot both be True")
        if self.S is None:
            return
        if not self.evaluated:
            self.populate_mesh()

        if not isinstance(band, (list, tuple, np.ndarray)):
            bands = [band]
        else:
            bands = band
        
        states = [self.states[..., band].reshape((self.mesh_points, np.prod(self.N[self.open_dim]), self.model.bands)) for band in bands]
        spin = [np.tensordot(self.S, np.swapaxes(np.conj(state), -1, -2) @ state, ([1, 2], [self.eff_dim, self.eff_dim + 1])).transpose().real for state in states]
        
        figs = []
        if subplots:
            fig, axs = plt.subplots(1, 3)
            fig.set_figwidth(12)
            fig.set_figheight(4)
        for i in range(3):
            if not subplots:
                fig = plt.figure()
                ax = fig.gca()
            else:
                ax = axs[i]

            dims = [i for i in range(self.model.dim) if i not in self.open_dim]
            dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)

            for b, s in zip(range(len(spin)), spin):
                ax.plot(self.mesh, s[:, i], label="$\\langle S\\rangle_{" + dim_labels(i) + "}^{" + str(bands[b] + 1) + "}$")

            ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
            ax.set_ylabel("$S_" + ["x", "y", "z"][i] + "(\\mathbf{k})$")
            ax.set_title("Spin expectations")
            ax.legend()
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])

            if not subplots:
                if save_fig:
                    if save_fig.endswith(".png"):
                        fig.savefig(save_fig[:-4] + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
                    else:
                        fig.savefig(save_fig + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
                else:
                    plt.show()
                if close_fig:
                    plt.close(fig)
                elif return_fig:
                    figs.append(fig)
        if subplots:
            fig.tight_layout()
            if save_fig:
                fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            elif return_fig:
                return fig
        elif return_fig:
            return figs

    def spin_heat_map_band(self, band, pi_ticks=True, close_fig=False, return_fig=False, save_fig="", max_magnitude: float=0.5, cmap="RdBu", subplots=False):
        """Sums over the bands for the spin expectation value."""
        max_magnitude = np.abs(max_magnitude)
        if self.eff_dim != 1 or len(self.open_dim) != 1:
            raise ValueError("Dimension of the model is not supported. Are boundaries opened correctly?")
        if save_fig:
            close_fig = True
        if close_fig and return_fig:
            raise ValueError("`close_fig` and `return_fig` cannot both be True")
        if self.S is None:
            return
        if not self.evaluated:
            self.populate_mesh()

        if not isinstance(band, (list, tuple, np.ndarray)):
            bands = [band]
        else:
            bands = band
        
        states = [self.states[..., band].reshape((self.mesh_points, np.prod(self.N[self.open_dim]), self.model.bands)) for band in bands]
        spin = [np.sum(np.dot(self.S, state.transpose((0, 2, 1))).transpose((0, 2, 1, 3)) * state.conj().transpose(0, 2, 1), axis=2).real for state in states]
        spin = sum(spin)

        figs = []
        if subplots:
            fig, axs = plt.subplots(1, 3)
            fig.set_figwidth(12)
            fig.set_figheight(4)
        for i in range(3):
            if not subplots:
                fig = plt.figure()
                ax = fig.gca()
            else:
                ax = axs[i]

            im = ax.imshow(spin[i].T, aspect=2 * np.pi / np.prod(self.N[self.open_dim]),
                           extent=(-np.pi, np.pi, 0, np.prod(self.N[self.open_dim])), origin="lower",
                           vmin=-max_magnitude, vmax=max_magnitude, cmap=cmap)
            if not subplots:
                fig.colorbar(im, ax=ax)

            
            dims = [i for i in range(self.model.dim) if i not in self.open_dim]
            dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)
            ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
            ax.set_ylabel("Site")
            ax.set_title("$\\langle S_%s\\rangle$ of bands %s" % (["x", "y", "z"][i], str(np.array(bands).astype(int) + 1)))

            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            if not subplots:
                if save_fig:
                    if save_fig.endswith(".png"):
                        fig.savefig(save_fig[:-4] + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
                    else:
                        fig.savefig(save_fig + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
                else:
                    plt.show()
                if close_fig:
                    plt.close(fig)
                elif return_fig:
                    figs.append(fig)
        if subplots:
            fig.tight_layout()
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            if save_fig:
                fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            elif return_fig:
                return fig
        elif return_fig:
            return figs

    def entanglement_spectrum(self, filled_bands=None, a_half=True, op: np.ndarray=None, trace: callable=None):
        if self.eff_dim != 1:
            print("Entanglement spectra is only supported in 1-D.")
            return
        if op is not None and trace is not None:
            print("Operator and trace cannot be specified together!")
            return
        
        filled_bands = filled_bands if filled_bands else self.eff_bands // 2
        if not self.evaluated:
            self.populate_mesh(filled_bands)
        # proj = np.zeros((self.mesh_points, self.eff_bands, self.eff_bands), dtype=np.complex64)
        if op is None:
            proj = self.gs_projector(filled_bands)
        else:
            op = np.kron(np.eye(np.prod(self.N[self.open_dim])), op)
            proj = op @ self.states[..., :, :filled_bands] @ np.swapaxes(np.conj(self.states[..., :, :filled_bands]), -1, -2) @ np.conj(op).T
        
        N = proj.shape[-1] // 2

        if a_half:
            proj = proj[..., :N, :N]
        else:
            proj = proj[..., N:, N:]
        
        if trace is not None:
            proj = trace(proj)
        w, _ = np.linalg.eigh(proj)
        return w
    
    def plot_entanglement_spectrum(self, filled_bands=None, pi_ticks=True, close_fig=False, return_fig=False, save_fig="", a_half=True, op=None, trace: callable=None):
        if save_fig:
            close_fig = True
        if close_fig and return_fig:
            raise ValueError("`close_fig` and `return_fig` cannot both be True")
        w = self.entanglement_spectrum(filled_bands, a_half=a_half, op=op, trace=trace)

        fig = plt.figure()
        ax = fig.gca()
        for i in range(w.shape[1]):
            ax.plot(self.mesh, w[:, i], "ko", markersize=3)

        dims = [i for i in range(self.model.dim) if i not in self.open_dim]
        dim_labels = lambda i: ["x", "y", "z", "w"][i] if i <= 3 else str(i)

        ax.set_xlabel("$k_{" + dim_labels(dims[0]) + "}$")
        ax.set_ylabel("$\\xi(k_" + dim_labels(dims[0]) + ")$")
        ax.set_title("Entanglement spectrum")
        if pi_ticks:
            ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
        if save_fig:
            fig.savefig(save_fig if save_fig.endswith(".png") else save_fig + ".png", dpi=600)
        else:
            plt.show()
        if close_fig:
            plt.close(fig)
        elif return_fig:
            return fig
