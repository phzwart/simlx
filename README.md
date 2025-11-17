# simlx

[![Release](https://img.shields.io/github/v/release/phzwart/simlx)](https://img.shields.io/github/v/release/phzwart/simlx)
[![Build status](https://img.shields.io/github/actions/workflow/status/phzwart/simlx/main.yml?branch=main)](https://github.com/phzwart/simlx/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/phzwart/simlx/branch/main/graph/badge.svg)](https://codecov.io/gh/phzwart/simlx)
[![Commit activity](https://img.shields.io/github/commit-activity/m/phzwart/simlx)](https://img.shields.io/github/commit-activity/m/phzwart/simlx)
[![License](https://img.shields.io/github/license/phzwart/simlx)](https://img.shields.io/github/license/phzwart/simlx)

self supervised representation learning

- **Github repository**: <https://github.com/phzwart/simlx/>
- **Documentation** <https://phzwart.github.io/simlx/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:phzwart/simlx.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Random Gaussian Projection Heads

Matryoshka U-Net now supports non-parameterized Gaussian projection heads that resample a batch-level projection matrix on every forward pass during training. Enable them with `MatryoshkaUNetConfig.use_random_projections=True` to obtain fast stochastic regularization (projection generation is roughly 1000× faster than QR-based alternatives).

```python
import torch
from simlx.models.matryoshka_unet import MatryoshkaUNet, MatryoshkaUNetConfig
from simlx.models.projection_utils import (
    analyze_projection_quality,
    compute_svd_projections,
    replace_random_projections,
)

# Stage 1: train with random projections
config = MatryoshkaUNetConfig(use_random_projections=True, in_channels=3, spatial_dims=2)
model = MatryoshkaUNet(config=config).train()

# Stage 2: optionally distill projections with SVD
svd_weights = compute_svd_projections(model, train_loader, device=torch.device("cuda"))
replace_random_projections(model, svd_weights)
model.eval()

# Stage 3: inspect projection quality
metrics = analyze_projection_quality(model, val_loader)
print(metrics["bottleneck"]["variance_explained"])
```

After calling `replace_random_projections`, the heads become deterministic and can be shipped alongside the trained weights (call `MatryoshkaUNet.eval()` before exporting).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/phzwart/simlx/settings/secrets/actions/new).
- Create a [new release](https://github.com/phzwart/simlx/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
