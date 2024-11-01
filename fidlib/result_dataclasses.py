"""We define dataclasses to be serialized to yaml to store the results of the experiments."""

from dataclasses import dataclass

from dataclass_wizard import YAMLWizard


@dataclass
class VarResult(YAMLWizard):
    r: list[float]
    qubits: list[int]
    num_parameters: list[int]
    t: list[float] | None
    variances: list[list[float]] | None
    infidelities: list[list[float]] | None
    lambdas: list[float]


@dataclass
class MovingMinimaResult(YAMLWizard):
    initial_parameters: list[float]
    cut_samples: list[float]
    infidelity: list[float]
    landscapes: list[list[float]]
    perturbation: list[list[float]]
    H: list[list[str, float]]
    h_norm: float
    t: list[float]
    seed: int


@dataclass
class MovingMinimaCompiledResult(YAMLWizard):
    qubits: int
    nseeds: int
    means: list[float]
    stds: list[float]
    mins: list[float]
    maxs: list[float]
    t: list[float]


@dataclass
class TildeMovingMinimaCompiledResult(YAMLWizard):
    qubits: list[int]
    nseeds: list[int]
    distances: list[list[list[float]]]
    t: list[float]
