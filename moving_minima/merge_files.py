"""Just a utility script to combine results from costly simulations."""

import os
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from fidlib.result_dataclasses import (MovingMinimaCompiledResult,
                                       MovingMinimaResult,
                                       TildeMovingMinimaCompiledResult)

directory = pathlib.Path(__file__).parents[1].resolve() / "moving_minima"
qubits = list(range(4, 14 + 1))
pert_sizes_list = [None] * len(qubits)
nseeds_list = [None] * len(qubits)


allfiles = os.listdir(directory / "XZ")

for k, n in tqdm(enumerate(qubits)):
    relevant_files = [
        filename
        for filename in allfiles
        if filename.startswith(f"moving_minima_qubits={n}")
    ]
    results = [
        MovingMinimaResult.from_yaml_file(directory / "XZ" / filename)
        for filename in tqdm(relevant_files)
    ]
    pert_sizes = [None] * len(results)

    for i, r in tqdm(enumerate(results)):
        pert_sizes[i] = np.array(
            [np.linalg.norm(c, np.inf) for c in r.perturbation]
        ).tolist()
    pert_sizes_list[k] = pert_sizes
    nseeds_list[k] = len(results)

TildeMovingMinimaCompiledResult(
    n,
    nseeds_list,
    pert_sizes_list,
    results[0].t,
).to_yaml_file(directory / f"averaged_minima.yaml")
