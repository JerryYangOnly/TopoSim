"""
Defines common constants used throughout the package.
Provides the Pauli matrices, Gell-Mann matrices,
the SU(4) generators, and spin representations.
"""
import numpy as np

# Define Pauli matrices
pauli = np.zeros((4, 2, 2), dtype=np.complex64)
pauli[0] = np.eye(2)
pauli[1] = np.array([[0, 1], [1, 0]])
pauli[2] = np.array([[0, -1j], [1j, 0]])
pauli[3] = np.array([[1, 0], [0, -1]])

# Define Gell-Mann matrices
gellmann = np.zeros((9, 3, 3), dtype=np.complex64)
gellmann[0] = np.eye(3)
gellmann[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
gellmann[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
gellmann[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
gellmann[4] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
gellmann[5] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
gellmann[6] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
gellmann[7] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
gellmann[8] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)

# Define su(4) generators
su4gen = np.zeros((16, 4, 4), dtype=np.complex64)
su4gen[0] = np.eye(4)
for i in range(1, 9):
    su4gen[i, :3, :3] = gellmann[i]
del i
su4gen[9] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
su4gen[10] = np.array([[0, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 0]])
su4gen[11] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
su4gen[12] = np.array([[0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 0, 0], [0, 1j, 0, 0]])
su4gen[13] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
su4gen[14] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
su4gen[15] = np.diag([1, 1, 1, -3]) / np.sqrt(6)

# Define spin matrices
def spin_matrices(spin : float):
    """Generates the matrix representation of
    quantum angular momentum operators.

    Parameters
    ----------
    spin : float
        The spin for which operators are generated.
        Must be a non-negative integer or half-integer.

    Returns
    -------
    sop : np.ndarray
        The corresponding angular momentum operators.
        Has shape (3, 2*spin+1, 2*spin+1), with the first axis
        corresponding to the x, y, z components.

    Raises
    ------
    ValueError
        If `spin` is not a non-negative integer or half integer.
    """
    if spin >= 0 and (int(spin) == spin or int(spin * 2) == spin * 2):
        size = int(2 * spin + 1)
        sop = np.zeros((3, size, size), dtype=np.complex64)
        m_z = np.arange(size - 1) - spin
        diag = np.sqrt(spin*(spin+1) - m_z*(m_z+1)) / 2
        sop[0] += np.diag(diag, k=1)
        sop[0] += np.diag(diag, k=-1)
        sop[1] += np.diag(diag * -1j, k=1)
        sop[1] += np.diag(diag * 1j, k=-1)
        sop[2] = np.diag(spin - np.arange(size))
        return sop
    raise ValueError("Only integer or half-integer spins accepted.")
