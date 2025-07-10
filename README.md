<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# <code>❯ U-Net Amazon Segmentation</code>

<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/segmentation_models_pytorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="segmentation_models_pytorch">
<img src="https://img.shields.io/badge/GeoPandas-139C5A.svg?style=default&logo=GeoPandas&logoColor=white" alt="GeoPandas">
<img src="https://img.shields.io/badge/rasterio-5B8C5A.svg?style=default&logo=Python&logoColor=white" alt="rasterio">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=default&logo=Pydantic&logoColor=white" alt="Pydantic">
<br>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/albumentations-FF6B00.svg?style=default&logo=Python&logoColor=white" alt="albumentations">
<img src="https://img.shields.io/badge/torchmetrics-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="torchmetrics">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [License](#license)

---

## Overview

This repository stores an application to a binary segmentation task of forest in Sentinel-2A Sattelite Images using an U-Net Model.


---

## Project Structure

```sh
└── /
    ├── config
    │   └── config.py
	├── docs
    ├── data
    │   ├── dataset.py
    │   ├── preprocessing.py
    │   └── transform.py
	├── model
    │   ├── loss.py
    │   └── model.py
    └── training
        ├── experiment.py
        ├── metrics.py
        └── training.py
	├── evaluation
    │   └── evaluation.py
	├── results
    ├── run_first_experiment.py
    ├── run_image_preprocessing.py
    ├── run_second_experiment.py
```
---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build  from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd 
    ```

3. **Install the dependencies:**

	```sh
	❯ pip install -r requirements.txt
	```

---

## License

 is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
