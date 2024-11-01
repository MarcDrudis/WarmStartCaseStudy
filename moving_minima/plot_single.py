"""Generates Fig. 4"""

import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from fidlib.result_dataclasses import (MovingMinimaResult,
                                       TildeMovingMinimaCompiledResult)

directory = pathlib.Path(__file__).parents[0].resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")

# Geta data
result_individual = MovingMinimaResult.from_yaml_file(
    directory / "XX" / "moving_minima_qubits=10_seed=0.yaml"
)

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(min(result_individual.t), max(result_individual.t))
line_colors = cmap(norm(result_individual.t))[::-1]

width_document = 510 / 72.27 / 2
# Create subplots
fig, ax = plt.subplots(1, 1, figsize=(width_document, width_document / 1.5))

# Plot Landscape
count = 0
relevant_times = [0, 10, 20]
for l, t_index, c in zip(
    result_individual.landscapes, range(len(result_individual.t)), line_colors
):
    t = np.round(result_individual.t[t_index], 5)
    ax.plot(
        np.array(result_individual.cut_samples) / np.pi,
        l,
        color=c,
        linestyle="-" if t_index not in relevant_times else "-.",
        linewidth=1.5 if t_index in relevant_times else 1,
        alpha=1 if t_index in relevant_times else 0.5,
        label=(rf"$\delta t={t}$" if t_index in relevant_times else None),
    )
ax.set_xlabel(r"Update Size, $\pi \norm{\bm{\theta}}_{\infty}$")
ax.tick_params(axis="x", labelsize=11)
ax.set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
ax.legend(
    borderpad=0.00001,
    handlelength=1.3,
    handletextpad=0.001,
    borderaxespad=0.4,
    loc="center left",
)

plt.savefig(directory.parent / f"plots/minima_jump.svg")
plt.savefig(directory.parent / f"plots/minima_jump.png")
