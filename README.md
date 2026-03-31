<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Possible Installation
```
mamba create -n ENV_NAME 'tensorflow-gpu>=2.16' --override-channels -c conda-forge
mamba activate ENV_NAME
mamba install llvm
pip install numpy-stl trimesh
pip install -e .
```
