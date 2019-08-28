# Contributing to Ignite

If you are interested in contributing to Ignite, your contributions will fall
into two categories:
1. You want to propose a new feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/pytorch/ignite/issues.
    - Pick an issue and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

In both cases, you will also need to code some tests to ensure the correct behaviour. We are using 
`pytest` to write our tests:
  - naming convention for files `test_*.py`, e.g. `test_precision.py`
  - naming of testing functions `def test_*`, e.g. `def test_precision_on_random_data()`

New code should be compatible with Python 2.7 and Python 3.X versions. Once you finish implementing a feature or bugfix and tests, please run lint checking and tests:

#### Run lint checking
```bash
flake8 ignite/ tests/
```

#### Run tests:
```bash
pytest tests/
```
or tests with coverage (assuming installed `pytest-cov`):
```bash
py.test --cov ignite --cov-report term-missing
```

We suggest to run test on Python 3.X, however be aware that our CI system is testing on Python 2.7 and Python 3.6. 

#### Send a PR
If everything is OK, please send a Pull Request to https://github.com/pytorch/ignite


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Writing documentation

Ignite uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.
