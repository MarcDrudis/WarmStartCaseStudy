from setuptools import find_packages, setup

setup(
    name="fidlib",
    packages=find_packages(),
    version="0.1.0",
    description="Python Package for Fidelity Landscape",
    author="Marc Sanz Drudis",
    license="MIT",
    install_requires=[
        "qiskit>=1",
        "qiskit-algorithms",
        "tqdm",
        "matplotlib",
        "plotly",
        "pandas",
        "pylatexenc",
        "nbformat",
        "numba",
        "black",
        "isort",
        "joblib",
        "joblib-progress",
        "seaborn",
        "dataclass-wizard[yaml]",
        "orqviz",
    ],
    dependency_links=["git+"],
)
