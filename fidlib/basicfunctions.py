"""Utility functions"""

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
    """Computes the eigenvalue of a Hamiltonian."""
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
    """
    Returns a Hamiltonian variational ansatz. That is a circuit with generators corresponding
    to the terms in the Hamiltonian.
    """
    qc = QuantumCircuit(H.num_qubits)

    if isinstance(H, SparsePauliOp):
        for i, pauli in enumerate(list(H.paulis) * reps):
            evolve_pauli(qc, pauli, time=Parameter(f"θ[{i}]") / 2, cx_structure="chain")
    elif isinstance(H, Pauli):
        evolve_pauli(qc, H, time=Parameter(f"θ") / 2, cx_structure="chain"),
    else:
        raise ValueError("H has to be either SparsePauliOp or Pauli.")

    return qc


def lattice_hamiltonian(
    num_qubits: int,
    terms: list[tuple[str, float]],
    periodic: bool = False,
) -> SparsePauliOp:
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
    """Returns connectivity of the Hamiltonian."""
    if periodic:
        connections = [
            np.arange(i, i + len(term)) % num_qubits for i in range(num_qubits)
        ]
    else:
        connections = [
            np.arange(i, i + len(term)) for i in range(num_qubits - len(term) + 1)
        ]
    return connections


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
