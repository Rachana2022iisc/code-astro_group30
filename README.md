# StellaClass

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20933979.svg)](https://doi.org/10.5281/zenodo.20933979) [![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)

## Motivation
Given a patch of sky, i.e. the RA/DEC, the package will import the sources from SDSS database, and will predict the class of the object. The prediction is based on a Supervised ML model using DecisionTree. For the purpose of using the package, one needs the .joblib file that contains the information of the trained model. You can download it directly from [Zenodo](https://doi.org/10.5281/zenodo.20933979), along with the training data. The file is supposed to be placed in the root directory of the cloned repository. If you are interested, you can modify the code and re-train the model.

## Installation
The package is installable on Python 3.x. To install the package, simply write:

`pip install StellaClass`

Otherwise, clone this repo, and follow the below specified commands:

`cd StellaClass`

`pip install -e .`

 A list of dependencies is available in requirements.txt.

 **Comment:** Creating a virtual environment and installing `python==3.10.20`, `scikit-learn==1.2.2` and `numpy==1.26.4` works flawlessly.

## Citation

This project uses the dataset from [Predicting Stellar Class](www.kaggle.com/competitions/playground-series-s6e6/overview). If you use this work, please also cite the original dataset:

> Author, Y. Yan, W. Reade, E. Park (2026). *train.csv* [Data set]. https://kaggle.com/competitions/playground-series-s6e6

```bibtex
@misc{playground-series-s6e6,
    author = {Yao Yan, Walter Reade, Elizabeth Park},
    title = {Predicting Stellar Class},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s6e6}},
    note = {Kaggle}
}
```
