<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sionna RT Reflectivity

Ray-based modeling of electromagnetic scattering and blockage by objects at radio frequencies, using geometrical optics (GO) and the Uniform Theory of Diffraction (UTD). This project extends the TensorFlow-based Sionna RT v0 codebase for wireless propagation and integrated sensing and communication (ISAC).

Curved objects are represented by triangular planar facets. The diffraction geometry supports both convex and concave edges. The choice of discretization affects the predicted electromagnetic response.

The project considers both backscattering and forward scattering, including the shadow/blockage region. The modeling approach and its validation for cylinders, spheres, and vehicles are described in [1](#references).

## Diffraction extensions

**Vertex diffraction.** Standard wedge UTD is derived for infinitely long edges. When applied to finite edges, the edge-diffracted ray alone introduces discontinuities where its diffraction point reaches an endpoint. Vertex UTD adds the endpoint contributions needed to obtain a continuous combined field through these transitions [2](#references). Faceted representations often contain finite edges with lengths comparable to the wavelength, making vertex diffraction essential for modeling their response.

**Double edge diffraction.** Applying the single-edge UTD coefficient twice is insufficient in overlapping transition regions. A uniform double-diffraction formulation requires a joint transition function and slope contributions that can become comparable to the leading contribution in these regions [3](#references).

**Mixed diffraction.** Additional edge–vertex and vertex–edge interactions extend the double-diffraction model. These include configurable approximations for the shadow region, as discussed in [1](#references).

## Installation

Use a separate Python 3.11 environment: this modified distribution retains the package and import name `sionna`.

For example, with Mamba on Linux:

```bash
mamba create -n sionna_reflectivity -c conda-forge python=3.11 pip llvm=23
mamba activate sionna_reflectivity
git clone https://github.com/AinurZiga/sionna-RT-reflectivity.git
cd sionna-RT-reflectivity
python -m pip install -e .
```

Python dependencies are specified in `setup.cfg`. CPU execution requires LLVM; GPU execution requires a working CUDA environment. The example notebooks contain an optional LLVM configuration block for installations that need an explicit library path or encounter a TensorFlow/LLVM conflict.

## Examples

Examples are provided in [`examples/scattering`](examples/scattering):

- [`cylinders.ipynb`](examples/scattering/cylinders.ipynb): scattering and shadow fields for a faceted cylinder, illustrating the effect of double edge diffraction.
- [`spheres.ipynb`](examples/scattering/spheres.ipynb): scattering from a discretized sphere, where vertex diffraction is particularly important, with full-wave comparisons.
- [`low_poly_car.ipynb`](examples/scattering/low_poly_car.ipynb): full-wave comparisons for a simplified vehicle in both the backscattering and shadow/blockage regions.

## References

1. A. Ziganshin, E. M. Vitucci, W. Kotterman, R. Thomä, C. Schneider, and V. Degli-Esposti, “Ray-Based Simulation of Scattering from Discretized Curved Bodies for Vehicular and ISAC Applications,” 2026. [arXiv:2604.05991](https://arxiv.org/abs/2604.05991).
2. M. Albani, F. Capolino, G. Carluccio, and S. Maci, “UTD Vertex Diffraction Coefficient for the Scattering by Perfectly Conducting Faceted Structures,” *IEEE Transactions on Antennas and Propagation*, vol. 57, no. 12, pp. 3911–3925, 2009. [DOI](https://doi.org/10.1109/TAP.2009.2027455).
3. M. Albani, “A Uniform Double Diffraction Coefficient for a Pair of Wedges in Arbitrary Configuration,” *IEEE Transactions on Antennas and Propagation*, vol. 53, no. 2, pp. 702–710, 2005. [DOI](https://doi.org/10.1109/TAP.2004.841289).

## License

Based on [Sionna](https://github.com/NVlabs/sionna) by NVIDIA and the Sionna contributors. Distributed under the [Apache License 2.0](LICENSE).
