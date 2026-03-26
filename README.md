# dreams 

A differentiable T-matrix-based framework for
multiple-scattering problems in nanophotonics.

It implements the T-matrix formalism in
[jax](https://github.com/google/jax), following the implementation of
[treams](https://github.com/tfp-photonics/treams).

## Installation

To install the package with pip, use

```bash
pip install dreams
```

For the development version, clone the repository:

```bash
git clone https://github.com/tfp-photonics/dreams.git
cd dreams
pip install -e ".[dev]"
```

## Usage

The tutorial is available in the documentation:
https://tfp-photonics.github.io/dreams/


## Features

- inverse design of multiple-scattering systems
- optimization of scatterer positions and radii
- support for finite and periodic arrangements
