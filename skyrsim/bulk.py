import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from .common import *
from .model import *

# A class for simulating results
class Simulator:
    def __init__(self, model: Model, mesh_points: int):
        self.model = model
        self.set_mesh(mesh_points)
        self.set_spin_op(None)

    def set_mesh(self, mesh_points):
        self.mesh_points = mesh_points
        self.mesh = np.linspace(-np.pi, np.pi, mesh_points) #, endpoint=False)
        # self.meshX, self.meshY = np.meshgrid(np.linspace(-np.pi, np.pi, mesh_points, endpoint=False))
        self.evaluated = False
        self.band = np.zeros((*([mesh_points] * self.model.dim), self.model.bands))
        self.states = np.zeros((*([mesh_points] * self.model.dim), self.model.bands, self.model.bands), dtype=np.complex64)

    def populate_mesh(self):
        if self.evaluated:
            return False
        # print("Starting evaluation")
        self.band = self.band.reshape(self.mesh_points**self.model.dim, self.model.bands)
        self.states = self.states.reshape(self.mesh_points**self.model.dim, self.model.bands, self.model.bands)
        for i in range(self.mesh_points**self.model.dim):
            idx = [0] * self.model.dim
            k = i
            for j in range(self.model.dim - 1, -1, -1):
                idx[j] = k % self.mesh_points
                k //= self.mesh_points
            w, v = scipy.linalg.eigh(self.model.hamiltonian([self.mesh[j] for j in idx]))
            self.band[i, :] = w
            self.states[i, :, :] = v
        self.band = self.band.reshape((*([self.mesh_points] * self.model.dim), self.model.bands))
        self.states = self.states.reshape((*([self.mesh_points] * self.model.dim), self.model.bands, self.model.bands))

        # Gauge fixing on the boundaries
        for i in range(self.model.dim):
            idx = [slice(None)] * (self.model.dim + 2)
            idx[i] = self.mesh_points - 1
            self.states[tuple(idx)] = self.states.take(0, axis=i)

        self.evaluated = True
        return True

    def gs_projector(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        return self.states[..., :, :filled_bands] @ np.swapaxes(np.conj(self.states[..., :, :filled_bands]), -1, -2)


    def direct_band_gap(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling
        return np.min(self.band[:, :, filled_bands] - self.band[:, :, filled_bands - 1])

    def indirect_band_gap(self, filled_bands=None):
        if not self.evaluated:
            self.populate_mesh()
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        return np.min(self.band[:, :, filled_bands]) - np.max(self.band[:, :, filled_bands - 1])

    def has_indirect_band_gap(self, filled_bands=None):
        return np.heaviside(self.indirect_band_gap(filled_bands), 0.0)

    def plot_band(self, filled_bands=None, full=False, pi_ticks=True, close_fig=True, save_fig=""):
        if not self.evaluated:
            self.populate_mesh()
        if not filled_bands:
            filled_bands = self.model.bands // 2    # Default to half-filling

        if self.model.dim == 1:
            fig, ax = plt.subplots()

            if not full:
                ax.plot(self.mesh, self.band[:, filled_bands - 1])
                ax.plot(self.mesh, self.band[:, filled_bands])
            else:
                for i in range(self.model.bands):
                    ax.plot(self.mesh, self.band[:, i])

            ax.set_xlabel("$k_x$")
            ax.set_ylabel("$E(k_x)$")
            ax.set_title("Band spectrum")
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            if save_fig:
                fig.savefig(save_fig, dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            else:
                return fig

        elif self.model.dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            X, Y = np.meshgrid(self.mesh, self.mesh)

            if not full:
                ax.plot_surface(X, Y, self.band[:, :, filled_bands - 1])
                ax.plot_surface(X, Y, self.band[:, :, filled_bands])
            else:
                for i in range(self.model.bands):
                    ax.plot_surface(X, Y, self.band[:, :, i])
            del X, Y

            ax.set_xlabel("$k_x$")
            ax.set_ylabel("$k_y$")
            ax.set_zlabel("$E(\\mathbf{k})$")
            ax.set_title("Band spectrum")
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
                ax.set_yticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            if save_fig:
                fig.savefig(save_fig, dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            else:
                return fig

        else:
            print("Band plotting of models in %d-D is not supported." % self.model.dim)


    def compute_chern(self, filled_bands=None, method="hatsugai", **kwargs):
        if self.model.dim != 2:
            print("Computation of Chern numbers is only supported in 2-D.")
            return
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        if method == "hatsugai":
            if not self.evaluated:
                self.populate_mesh()
            return self._chern_hatsugai(filled_bands)
        elif method == "hwcc":
            return self._chern_wcc(filled_bands, **kwargs)

    def _chern_hatsugai(self, filled_bands):
        Q = 0.0
        F = np.conj(self.states[:-1, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[1:, :-1, :, :filled_bands]
        F = F @ (np.conj(self.states[1:, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[1:, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[1:, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[:-1, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[:-1, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[:-1, :-1, :, :filled_bands])
        F = np.angle(np.linalg.det(F))
        Q = np.sum(F) / 2.0 / np.pi
        return Q
    
    def _chern_wcc(self, filled_bands, wl_density=100, axis=0, force_density=False):
        if (wl_density >= self.mesh_points - 1) or not self.evaluated or force_density:
            p = np.zeros((wl_density, filled_bands))

            for i, ky in zip(range(wl_density), np.linspace(-np.pi, np.pi, wl_density)):
                if axis == 0:
                    p[i] = self.wilson_loop(lambda x: ((x - 1/2) * 2 * np.pi, ky), wl_density, filled_bands=filled_bands, phases=True) / 2 / np.pi
                else:
                    p[i] = self.wilson_loop(lambda x: (ky, (x - 1/2) * 2 * np.pi), wl_density, filled_bands=filled_bands, phases=True) / 2 / np.pi
        else:
            wl_density = self.mesh_points - 1
            p = np.zeros((wl_density, filled_bands))

            for i in range(wl_density):
                p[i] = self._wilson_loop_grid((0, i) if axis == 0 else (i, 0), filled_bands=filled_bands, axis=axis) / 2 / np.pi
        
        p = np.sum(p, axis=1)
        p -= np.floor(p)
        p -= np.roll(p, 1)
        p = p[1:]
        p[p > 0.5] -= 1
        p[p <= -0.5] += 1
        return np.sum(p)

    def compute_z2(self, filled_bands=None, SOC=True, method="hwcc", **kwargs):
        if self.model.dim != 2:
            raise ValueError("Computation of Z2 invariant is only supported in 2-D")

        if not filled_bands:
            filled_bands = self.model.bands // 2

        if not SOC:
            if filled_bands and filled_bands % 2 == 1:
                raise ValueError("`filled_bands` cannot be odd in Z2 computation")

            model_s1 = Model()
            model_s2 = Model()
            model_s1.hamiltonian = lambda k: self.model.hamiltonian(k)[:self.model.bands // 2, :self.model.bands // 2]
            model_s2.hamiltonian = lambda k: self.model.hamiltonian(k)[self.model.bands // 2:, self.model.bands // 2:]
            model_s1.dim = model_s2.dim = self.model.dim
            model_s1.bands = model_s2.bands = self.model.bands // 2
            sim_s1 = Simulator(model_s1, self.mesh_points)
            sim_s2 = Simulator(model_s2, self.mesh_points)
            v = ((sim_s1.compute_chern(filled_bands // 2) - sim_s2.compute_chern(filled_bands // 2)) / 2) % 2
            del model_s1, model_s2, sim_s1, sim_s2
            return v
        else:
            if method == "fu_kane":
                return self._z2_fu_kane(filled_bands)
            elif method == "hwcc":
                return self._z2_wcc(filled_bands, **kwargs)
            else:
                raise AttributeError("`method` %s is not available" % method)

    def _z2_fu_kane(self, filled_bands):
        # Method of Fu and Kane
        if self.mesh_points % 2 == 0:
            self.set_mesh(self.mesh_points + 1)
        if not self.evaluated:
            self.populate_mesh()

        # Compute Berry flux for half of the Brillouin zone
        Q = 0.0
        n = self.mesh_points // 2
        F = np.conj(self.states[n:-1, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n + 1:, :-1, :, :filled_bands]
        F = F @ (np.conj(self.states[n + 1:, :-1, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n + 1:, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[n + 1:, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n:-1, 1:, :, :filled_bands])
        F = F @ (np.conj(self.states[n:-1, 1:, :, :filled_bands]).transpose(0, 1, 3, 2) @ self.states[n:-1, :-1, :, :filled_bands])
        F = -np.angle(np.linalg.det(F))
        Q = np.sum(F)

        # Compute integral of Berry connection about edge of effective Brillouin zone
        def eff_bz_loop(x):
            v = 6 * np.pi
            if x <= 1 / 3:
                return (np.pi, -np.pi + v * x)
            elif x <= 1 / 2:
                return (np.pi - v * (x - 1 / 3), np.pi)
            elif x <= 5 / 6:
                return (0, np.pi - v * (x - 1 / 2))
            else:
                return (v * (x - 5 / 6), -np.pi)
        W = self.wilson_loop(eff_bz_loop, filled_bands=filled_bands)
        # print(np.angle(W))
        W = np.angle(scipy.linalg.det(W))
        
        return ((W - Q) / 2 / np.pi) % 2

    def _z2_wcc(self, filled_bands, wl_density=100, axis=0, force_density=False):
        if self.mesh_points % 2 == 0 or (wl_density >= self.mesh_points // 2 + 1) or not self.evaluated or force_density:
            x = np.zeros((wl_density, filled_bands))

            for i, ky in zip(range(wl_density), np.linspace(0, np.pi, wl_density)):
                if axis == 0:
                    x[i] = self.wilson_loop(lambda x: ((x - 1/2) * 2 * np.pi, ky), wl_density, filled_bands=filled_bands, phases=True) / 2 / np.pi
                else:
                    x[i] = self.wilson_loop(lambda x: (ky, (x - 1/2) * 2 * np.pi), wl_density, filled_bands=filled_bands, phases=True) / 2 / np.pi
        else:
            wl_density = self.mesh_points // 2 + 1
            x = np.zeros((wl_density, filled_bands))

            for i in range(wl_density):
                x[i] = self._wilson_loop_grid((0, i) if axis == 0 else (i, 0), filled_bands=filled_bands, axis=axis) / 2 / np.pi

        g = np.zeros(wl_density)
        x -= np.floor(x)

        for i in range(wl_density):
            s = np.sort(x[i])
            x[i, :] = s
            s -= np.roll(s, 1)
            s[0] += 1
            k = np.argmax(s)
            g[i] = (x[i, k] + x[i, k - 1]) / 2 if k != 0 else ((x[i, k] + x[i, k - 1] + 1) / 2) % 1

        # for i in range(filled_bands):
        #     plt.plot(x[:, i], np.linspace(0, np.pi, wl_density), "o")
        # plt.plot(g, np.linspace(0, np.pi, wl_density))
        # plt.show()

        n = 0
        for i in range(wl_density - 1):
            gmin, gmax = (g[i], g[i + 1]) if g[i] < g[i + 1] else (g[i + 1], g[i])
            n += np.sum((x[i] >= gmin) & (x[i] < gmax))

        return n % 2


    def wilson_loop(self, loop, points=100, filled_bands=None, phases=False):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        W = np.eye(self.model.bands, dtype=np.complex64)
        origin = None
        for p in np.linspace(0, 1, points, endpoint=False):
            w, v = scipy.linalg.eigh(self.model.hamiltonian(loop(p)))
            if p == 0.0:
                origin = v[:, :filled_bands]
            else:
                # P = sum([np.outer(v[:, i], np.conj(v[:, i])) for i in range(filled_bands)])
                # W = P @ W
                W = v[:, :filled_bands] @ v[:, :filled_bands].conj().T @ W
        W = np.conj(origin).T @ W @ origin
        return W if not phases else np.sort(np.angle(scipy.linalg.eig(W)[0]))# np.sort(np.angle(scipy.linalg.eig(self.wilson_loop(loop, points, filled_bands))[0]))

    def _wilson_loop_grid(self, index, filled_bands, axis=0):
        W = np.eye(self.model.bands, dtype=np.complex64)
        origin = None

        index = np.array(index)
        for i in range(self.mesh_points - 1):
            index[axis] = i
            v = self.states
            for j in index:
                v = v[j]
            v = v[:, :filled_bands]

            if i == 0:
                origin = v
            else:
                W = v @ v.conj().T @ W
        W = np.conj(origin).T @ W @ origin
        return np.sort(np.angle(scipy.linalg.eig(W)[0]))
    

    def set_spin_op(self, S):
        self.S = S
        self.spin_evaluated = 0
        self.spin = np.zeros((*([self.mesh_points] * self.model.dim), 3))

    def populate_spin(self, filled_bands=None):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        if (self.spin_evaluated and self.spin_evaluated == filled_bands) or self.S is None:
            return
        if not self.evaluated:
            self.populate_mesh()
        self.spin = np.tensordot(self.S, np.conj(self.states[..., :filled_bands]) @ np.swapaxes(self.states[..., :filled_bands], -1, -2), ([1, 2], [self.model.dim, self.model.dim + 1])).transpose((1, 2, 0)).real
        self.spin_evaluted = filled_bands

    def minimum_spin_gap(self, filled_bands=None):
        filled_bands = filled_bands if filled_bands else self.model.bands // 2
        self.populate_spin(filled_bands)
        return np.min(np.sqrt(np.sum(self.spin**2, axis=2)))

    def normalized_spin(self):
        s = np.sqrt(np.sum(self.spin**2, axis=2).reshape((self.mesh_points, self.mesh_points, 1)))
        s[s == 0] = np.finfo(np.float32).eps
        return self.spin / s

    def compute_skyrmion(self, filled_bands=None, method="hatsugai"):
        """S has shape (3, bands, bands).
        """
        if self.S is None:
            return
        if self.model.dim != 2:
            print("Computation of skyrmion number is only supported in 2-D.")
            return
        self.populate_spin(filled_bands)

        spin = self.normalized_spin()
        # Normalize the spin for the computation
        hamil = np.zeros((self.mesh_points, self.mesh_points, 2, 2), dtype=np.complex64)
        for i in range(3):
            hamil += pauli[i + 1] * spin[..., i, np.newaxis, np.newaxis]

        if method == "hatsugai":
            _, states = np.linalg.eigh(hamil)
            states[-1] = states[0]          # Gauge fixing
            states[:, -1] = states[:, 0]
            Q = 0.0
            F = np.conj(states[:-1, :-1, :, :1]).transpose(0, 1, 3, 2) @ states[1:, :-1, :, :1]
            F = F @ (np.conj(states[1:, :-1, :, :1]).transpose(0, 1, 3, 2) @ states[1:, 1:, :, :1])
            F = F @ (np.conj(states[1:, 1:, :, :1]).transpose(0, 1, 3, 2) @ states[:-1, 1:, :, :1])
            F = F @ (np.conj(states[:-1, 1:, :, :1]).transpose(0, 1, 3, 2) @ states[:-1, :-1, :, :1])
            F = np.angle(np.linalg.det(F))
            Q = np.sum(F) / 2.0 / np.pi
            return Q
        elif method == "integral":
            return -np.sum(spin * np.cross(np.gradient(spin, axis=0), np.gradient(spin, axis=1))) / 4 / np.pi
        else:
            raise AttributeError("`method` %s is not available" % method)

    def compute_skyrmion_z2(self, Ss, filled_bands=None, SOC=True):
        if self.model.dim != 2:
            raise ValueError("Computation of skyrmion Z2 invariant is only supported in 2-D")
        if not SOC:
            if filled_bands and filled_bands % 2 == 1:
                raise ValueError("`filled_bands` cannot be odd in Z2 computation")
            sop1 = np.kron(np.array([[1, 0], [0, 0]]), Ss)
            sop2 = np.kron(np.array([[0, 0], [0, 1]]), Ss)
            self.set_spin_op(sop1)
            v = self.compute_skyrmion(filled_bands)
            self.set_spin_op(sop2)
            v -= self.compute_skyrmion(filled_bands)
            self.spin_evaluated = 0

            def _smooth_round(vf, threshold=1e-4):
                rvf = np.round(vf)
                if np.abs(rvf - vf) > threshold:
                    return vf
                else:
                    return rvf

            return (_smooth_round(v) / 2) % 2
        else:
            raise NotImplementedError

    def plot_spin_texture(self, filled_bands=None, normalize=True, pi_ticks=True, close_fig=True, save_fig=""):
        if self.model.dim != 2:
            print("Spin texture plotting is only supported in 2-D.")
            return

        self.populate_spin(filled_bands)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        kx, ky, kz = np.meshgrid(self.mesh, self.mesh, np.zeros(1))

        s = self.spin if not normalize else self.normalized_spin()
        ax.quiver(kx, ky, kz, s[:, :, 0:1], s[:, :, 1:2], s[:, :, 2:3], length=2 * np.pi / self.mesh_points, arrow_length_ratio=0.2)
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")
        ax.set_zlabel("$k_z$")
        ax.set_title("Spin texture")
        if pi_ticks:
            ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            ax.set_yticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
        ax.set_zlim(-np.pi, np.pi)
        if save_fig:
            fig.savefig(save_fig, dpi=600)
        else:
            plt.show()
        if close_fig:
            plt.close(fig)
        else:
            return fig

    def plot_spin_heat_map(self, filled_bands=None, normalize=True, pi_ticks=True, close_fig=True, save_fig=""):
        if self.model.dim != 2:
            print("Spin texture plotting is only supported in 2-D.")
            return

        self.populate_spin(filled_bands)
        s = self.spin if not normalize else self.normalized_spin()

        figs = []
        for i in range(3):
            fig = plt.figure()
            ax = fig.gca()
            pos = ax.imshow(s[:, :, i].transpose(), cmap="coolwarm", extent=(-np.pi, np.pi, -np.pi, np.pi), origin="lower")
            ax.set_xlabel("$k_x$")
            ax.set_ylabel("$k_y$")
            ax.set_title("$\\langle S\\rangle_" + ["x", "y", "z"][i] + "$")
            if pi_ticks:
                ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
                ax.set_yticks(np.linspace(-np.pi, np.pi, 5), ["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            fig.colorbar(pos, ax=ax)
            if save_fig:
                if save_fig.endswith(".png"):
                    fig.savefig(save_fig[:-4] + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
                else:
                    fig.savefig(save_fig + ["_x.png", "_y.png", "_z.png"][i], dpi=600)
            else:
                plt.show()
            if close_fig:
                plt.close(fig)
            else:
                figs.append(fig)
        if not close_fig:
            return figs
