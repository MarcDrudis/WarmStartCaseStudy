import numba
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.sparse.linalg import expm_multiply


class VarianceComputer:
    def __init__(
        self,
        qc: QuantumCircuit,
        initial_parameters: np.ndarray | None,
        times: np.ndarray | tuple[float, int] | None,
        H: SparsePauliOp,
    ):
        self.qc = qc
        self.initial_parameters = (
            np.random.uniform(-np.pi, np.pi, qc.num_parameters)
            if initial_parameters is None
            else initial_parameters
        )
        if isinstance(times, tuple):
            self.evolved_states = expm_multiply(
                A=-1.0j * H.to_matrix(sparse=True),
                B=Statevector(qc.assign_parameters(self.initial_parameters)).data,
                start=0,
                stop=times[0],
                num=times[1],
            )

        elif times is None:
            s = Statevector(qc.assign_parameters(self.initial_parameters))
            l = [s]
            self.evolved_states = np.array(l)

        else:
            self.evolved_states = np.array(
                [
                    expm_multiply(
                        A=-1.0j * t * H.to_matrix(sparse=True),
                        B=Statevector(
                            qc.assign_parameters(self.initial_parameters)
                        ).data,
                    )
                    for t in times
                ]
            )

    def _sample_points(self, batch_size: int, omega: float) -> np.ndarray:
        return np.random.uniform(
            low=-omega, high=omega, size=(batch_size, self.qc.num_parameters)
        )

    def _compute_batch(self, batch_size: int, omega: float) -> np.ndarray:
        dthetas = self._sample_points(batch_size=batch_size, omega=omega)
        perturbed_states = np.array(
            [
                Statevector(
                    self.qc.assign_parameters(self.initial_parameters + dthetas[i])
                ).data
                for i in range(batch_size)
            ]
        )

        return _abs2(np.inner(perturbed_states, self.evolved_states.conj()))

    def compute_averages(
        self, batch_size: int, N_batches: int, omega: float
    ) -> tuple[float, float]:
        L = 0
        L2 = 0
        for _ in range(N_batches):
            batch = self._compute_batch(batch_size=batch_size, omega=omega)
            L += np.sum(batch, axis=0)
            L2 += np.sum(batch**2, axis=0)

        N_samples = N_batches * batch_size

        return L / N_samples, L2 / N_samples

    def compute_variance(self, batch_size: int, N_batches: int, omega: float) -> float:
        L, L2 = self.compute_averages(
            batch_size=batch_size, N_batches=N_batches, omega=omega
        )
        return L2 - L**2

    def direct_compute_variance(self, batch_size: int, omega: float) -> float:
        batch = self._compute_batch(batch_size=batch_size, omega=omega)
        return np.var(batch, axis=0)


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def _abs2(x):
    return x.real**2 + x.imag**2


def kplus(omega: float) -> float:
    """Computes 1/(2w) * Inegral(cos(x)^2,-omega,omega)"""
    return 0.5 + np.sin(2 * omega) / (4 * omega)


def cplus(omega: float) -> float:
    """Computes 1/(2w) * Inegral(cos(x)^4,-omega,omega)"""
    return (12 * omega + 8 * np.sin(2 * omega) + np.sin(4 * omega)) / (32 * omega)


def a_const(omega: float) -> float:
    "Computes cplus-kplus^2"
    return cplus(omega) - kplus(omega) ** 2


from scipy.optimize import minimize_scalar


def bound(omega: float, alpha: float, num_param: int) -> float:
    "Computes the full bound"
    fun = (
        lambda beta: a_const(omega)
        * (alpha * kplus(omega) ** num_param - beta * (1 - kplus(omega) ** num_param))
        ** 2
    )
    return minimize_scalar(fun, bounds=(-2, 2)).fun
