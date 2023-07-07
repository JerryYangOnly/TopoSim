import numpy as np
import scipy
import matplotlib.pyplot as plt

from model import *
from bulk import Simulator
from edge import EdgeSimulator

# chern = np.zeros(51)
# skyr = np.zeros(51)
# for i, u in zip(range(51), np.linspace(-3, 3, 51)):
#     sim = Simulator(FourBandModel(b=1.0, k=u), 21)
#     # print("u = %d, BG = %.2f" % (i, sim.direct_band_gap()))
#     # sim.plot_band()
#     # print(sim.compute_chern())
#     chern[i] = sim.compute_chern()
#     skyr[i] = sim.compute_skyrmion((np.kron(pauli[3], pauli[1]), np.kron(pauli[0], pauli[2]), np.kron(pauli[3], pauli[3])))
#     del sim
# plt.plot(np.linspace(-3, 3, 51), chern, "o-")
# plt.title("Chern number of four band model")
# plt.xlabel("k")
# plt.ylabel("C")
# plt.show()
# plt.plot(np.linspace(-3, 3, 51), skyr, "o-")
# plt.title("Skyrmion number of four band model")
# plt.xlabel("u")
# plt.ylabel("Q")
# plt.show()

# z2 = np.zeros(30)
# for i, u in zip(range(30), np.linspace(-3, 3, 30)):
#     sim = Simulator(BHZModel(u=u), 21)
#     # print("u = %.2f, BG = %.2f" % (i, sim.direct_band_gap()))
#     z2[i] = sim.compute_z2(SOC=True)
#     print("u = %.2f, Z2 = %.3f" % (u, z2[i]))
#     # phases = np.zeros((2, 100))
#     # for i, ky in zip(range(100), np.linspace(-np.pi, np.pi, 100)):
#     #     p = sim.wilson_loop(lambda x: (2*np.pi*(x - 1/2), ky), phases=True)
#     #     phases[:, i] = p
#     # plt.plot(phases[0], np.linspace(-np.pi, np.pi, 100), "k-")
#     # plt.plot(phases[1], np.linspace(-np.pi, np.pi, 100), "k-")
#     # plt.xlim(-np.pi, np.pi)
#     # plt.show()
#     del sim
# plt.plot(np.linspace(-3, 3, 30), np.round(z2) % 2)
# plt.show()

for u in np.arange(-2.5, 2.6, 1):
    sim = EdgeSimulator(BHZModel(u=u, SOC=0.1*pauli[2]), 41)
    sim.open((10, 0))
    sim.plot_band(full=True)
    states = sim.pdf(sim.in_gap_states(), sum_internal=True)
    for state in states:
        plt.bar(np.arange(10), state)
        plt.show()
