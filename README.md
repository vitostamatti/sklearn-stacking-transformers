# Sklearn Stacking Transformer


[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/vitostamatti/sklearn-stacking-transformers.svg)](https://github.com/vitostamatti/sklearn-stacking-transformers/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/vitostamatti/sklearn-stacking-transformers.svg)](https://github.com/vitostamatti/sklearn-stacking-transformers/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)



## 📝 Table of Contents

- [About](#about)
- [Setup](#setup)
- [Usage](#usage)


## About <a name = "about"></a>

Implementation of a scikit-learn transformers for doing stacking of models.
Parallel processes where use when possible.
This small package provides two transformers:

- ``RegressionStackingTransformer``
- ``ClassificationStackingTransformer``



## Setup <a name = "setup"></a>

To get started, clone this repo and check that you have all requirements installed.

```
git clone https://github.com/vitostamatti/sklearn-stacking-transformers.git
pip install .
``` 

## Usage <a name = "usage"></a>

Basic usage of this objects can be on a preprocessing pipeline.

```
from stacking_transformer import RegressionStackingTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

estimators = [
    ('et', ExtraTreesRegressor(
            random_state=0,
            n_jobs=-1,
            n_estimators=100,
            max_depth=3
    )),
    ('rf', RandomForestRegressor(
            random_state=0,
            n_jobs=-1,
            n_estimators=100,
            max_depth=3
    )),
    ('knn',KNeighborsRegressor(n_neighbors=10))
]

stack = RegressionStackingTransformer(
        estimators=estimators_l1,
        shuffle=True,
        random_state=0,
        verbose=1,
        n_jobs=-1
)

pipeline = Pipeline([
        ('stack',stack), 
        ("lr",LinearRegression())
])

```


In the [notebooks](/notebooks/) directory you can find examples of
usage for each object in this repo.

- [Regression](/notebooks/stacking_regression_example.ipynb) 
- [Classification](/notebooks/stacking_regression_example.ipynb) 


You can also take a look of the source code [here](/src/stacking_transformer.py)


## Roadmap

- [X] First commit of sklearn-stacking-transformers.
- [ ] Clean up RegressionStackingTransformer
- [ ] Finish ClassificationStackingTransformer.
- [ ] Complete documentation of examples notebooks.



## License

[MIT](LICENSE.txt)