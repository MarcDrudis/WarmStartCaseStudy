from typing import Callable

import numpy as np
import plotly.express as px
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import (Pauli, SparsePauliOp, Statevector,
                                 state_fidelity)
from qiskit.synthesis.evolution.product_formula import evolve_pauli
from qiskit_algorithms import SciPyRealEvolver, TimeEvolutionProblem
from scipy.interpolate import make_interp_spline
from scipy.optimize import approx_fprime
from scipy.sparse.linalg import eigsh


def get_ansatz(num_qubits: int, depth: str):
    """
    Creates an ansatz with a given number of qubits and a depth that scales
    either linearly or is constant with respect to number of qubits.
    """
    if depth not in ("linear", "const"):
        raise ValueError("Depth must be either 'linear' of 'const' ")
    reps = 6 if depth == "const" else max(num_qubits - 1, 1)
    return EfficientSU2(num_qubits=num_qubits, reps=reps)


def find_local_minima(
    fun: Callable,
    x0: np.ndarray,
    *args,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Find the closest local minima using slow gradient descent.

    Parameters:
    - fun: The target function.
    - x0: Initial point.
    - *args: Extra arguments for the target function.
    - learning_rate: Step size for gradient descent.
    - max_iterations: Maximum number of iterations.
    - epsilon: Convergence threshold.

    Returns:
    - Local minima as a numpy array.
    """
    x = x0.copy()

    for _ in range(max_iterations):
        gradient = approx_fprime(x, fun, 1.49e-8, *args)
        x -= learning_rate * gradient
        if np.linalg.norm(gradient) < epsilon:
            break

    return x


def eigenrange(H: SparsePauliOp) -> float:
    if isinstance(H, SparsePauliOp):
        H = H.to_matrix(sparse=True)
        smallest = eigsh(H, which="SA", k=1)[0][0]
        biggest = eigsh(H, which="LA", k=1)[0][0]
    else:
        raise ValueError(
            "The current Hamiltonian type is not supported. Hamiltonian is", type(H)
        )

    return biggest - smallest


def ansatz_QRTE_Hamiltonian(H: SparsePauliOp | Pauli, reps: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(H.num_qubits)

    if isinstance(H, SparsePauliOp):
        for i, pauli in enumerate(list(H.paulis) * reps):
            evolve_pauli(qc, pauli, time=Parameter(f"θ[{i}]") / 2, cx_structure="chain")
    elif isinstance(H, Pauli):
        evolve_pauli(qc, H, time=Parameter(f"θ") / 2, cx_structure="chain"),
    else:
        raise ValueError("H has to be either SparsePauliOp or Pauli.")

    return qc


# def qubit_variance(
#     num_qubits: int, r: float, depth: str, samples: int, hamiltonian: SparsePauliOp
# ) -> float:
#     """
#     Computes the variance for a given quantum circuit perturbed by the
#     time evolution of a given Hamiltonian.
#     Args:
#         num_qubits(int): number of qubits of the system
#         r(float): side of the hypercube to sample from
#         depth(str): "linear" or "const" for the number of repetitions
#         of the ansatz
#     """
#     qc = get_ansatz(num_qubits, depth)
#     times = None
#     vc = VarianceComputer(
#         qc=qc,
#         initial_parameters=None,
#         times=times,
#         H=hamiltonian,
#     )
#     return vc.direct_compute_variance(samples, r)


def bis_lattice_hamiltonian(
    num_qubits: int,
    terms: list[tuple[str, float]],
    periodic: bool = False,
):
    one_local_connections = [(c,) for c in range(num_qubits)]
    two_local_connections = [
        (cA, cB) for cA, cB in zip(range(num_qubits - 1), range(1, num_qubits))
    ]
    if periodic:
        two_local_connections += [(0, num_qubits - 1)]

    all_terms_list = []
    for term, coeff in terms:
        connections = one_local_connections if len(term) == 1 else two_local_connections
        all_terms_list += [(term, c, coeff) for c in connections]

    H = SparsePauliOp.from_sparse_list(all_terms_list, num_qubits=num_qubits)

    return H


def lattice_hamiltonian(
    num_qubits: int,
    terms: list[tuple[str, float]],
    periodic: bool = False,
):
    """
    Returns a Hamiltonian in a 1D lattice.
    Args:
    num_qubits: Number of qubits in the system.
    terms:Terms to be repited over the lattice.
    periodic: If true, use periodic boundary conditions
    """
    all_terms_list = []
    for term, coeff in terms:
        all_terms_list += [
            (term, c, coeff) for c in _n_local_connections(num_qubits, term, periodic)
        ]

    H = SparsePauliOp.from_sparse_list(all_terms_list, num_qubits=num_qubits)

    return H


def _n_local_connections(num_qubits: int, term: str, periodic: bool):
    if periodic:
        connections = [
            np.arange(i, i + len(term)) % num_qubits for i in range(num_qubits)
        ]
    else:
        connections = [
            np.arange(i, i + len(term)) for i in range(num_qubits - len(term) + 1)
        ]
    return connections


# def create_heisenberg(
#     num_qubits: int, j_const: float, g_const: float, circular: bool = False
# ) -> SparsePauliOp:
#     """Creates an Heisenberg Hamiltonian on a lattice."""
#     xx_op = ["I" * i + "XX" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
#     yy_op = ["I" * i + "YY" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
#     zz_op = ["I" * i + "ZZ" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
#
#     circ_op = (
#         ["X" + "I" * (num_qubits - 2) + "X"]
#         + ["Y" + "I" * (num_qubits - 2) + "Y"]
#         + ["Z" + "I" * (num_qubits - 2) + "Z"]
#         if circular
#         else []
#     )
#
#     z_op = ["I" * i + "Z" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]
#
#     return (
#         SparsePauliOp(xx_op + yy_op + zz_op + circ_op) * j_const
#         + SparsePauliOp(z_op) * g_const
#     )
#
#
# def create_ising(
#     num_qubits: int,
#     zz_const: float,
#     z_const: float,
#     circular: bool = False,
#     x_const: float = 0,
# ) -> SparsePauliOp:
#     """Creates an Heisenberg Hamiltonian on a lattice."""
#     zz_op = ["I" * i + "ZZ" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
#
#     circ_op = +["Z" + "I" * (num_qubits - 2) + "Z"] if circular else []
#
#     z_op = ["I" * i + "Z" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]
#     x_op = ["I" * i + "X" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]
#
#     return (
#         SparsePauliOp(zz_op + circ_op) * zz_const
#         + SparsePauliOp(z_op) * z_const
#         + SparsePauliOp(x_op) * x_const
#     )


def fidelity_var_bound(deltatheta, dt, m, eigenrange):
    return (0.5 * (1 + np.sinc(2 * deltatheta))) ** m * (
        1 - dt**2 / 4 * (eigenrange) ** 2
    )


def evolve_circuit(H, dt, qc, initial_parameters, observables, num_timesteps=100):
    prob = TimeEvolutionProblem(
        hamiltonian=H,
        time=dt,
        initial_state=qc.assign_parameters(initial_parameters),
        aux_operators=observables,
    )
    solver = SciPyRealEvolver(num_timesteps=num_timesteps)
    return solver.evolve(prob)


def find_maximum(x: list[float], y: list[float], n_around: int = 2):
    """This function just takes the x and y values of a function and
    interpolates the points closest to the maximum. This allows to infer
    the true value of the maximum even if we don't sample the exact x_value
    that it corresponds to."""
    max_index = np.argmax(y)
    x_interpolate = x[max_index - n_around : max_index + n_around + 1]
    y_interpolate = y[max_index - n_around : max_index + n_around + 1]
    x_range = np.linspace(min(x_interpolate), max(x_interpolate), 500)
    y_fun = make_interp_spline(x_interpolate, y_interpolate, k=3)(x_range)
    return x_range, y_fun
