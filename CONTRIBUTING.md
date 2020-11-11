# Contributing to Ignite

This project is a community effort, and everyone is welcome to contribute !

If you are interested in contributing to Ignite, there are many ways to help out. Your contributions may fall
into the following categories:

1. It helps us very much if you could 
    - Report issues you’re facing
    - Give a :+1: on issues that others reported and that are relevant to you
    - Spread a word about the project or simply :star: to say "I use it" 

2. Answering queries on the issue tracker, investigating bugs and reviewing other developers’ pull requests are 
very valuable contributions that decrease the burden on the project maintainers.

3. You would like to improve the documentation. This is no less important than improving the library itself! 
If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

4. You would like propose a new feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.

5. You would like implement a feature or bug-fix for an outstanding issue
    - Look at the issues labelled as 
["help wanted"](https://github.com/pytorch/ignite/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
    - Pick an issue and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.


## Developing Ignite

### Quickstart guide for first-time contributors
 
<summary>

<details>

- Install [miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your system.
- Create an isolated conda environment for pytorch-ignite:
 ```bash
conda create -n pytorch-ignite-dev python=3
```

- Activate the newly created environment:
 ```bash
conda activate pytorch-ignite-dev
```

- When developing please take care of preserving `.gitignore` file and make use of `.git/info/exclude` to exclude custom files like: `.idea`, `.vscode` etc.
- Please refer to [github first contributions guidelines](https://github.com/firstcontributions/first-contributions) and don't hesitate to ask the pytorch-ignite community in case of any doubt.
- A good way to start is to tackle one of the [good first issues](https://github.com/pytorch/ignite/labels/good%20first%20issue).

</details>

</summary>

### Installation

To get the development installation with the necessary dependencies, run the following:
```bash
git clone https://github.com/pytorch/ignite.git
cd ignite
python setup.py develop
pip install -r requirements-dev.txt
pip install flake8 "black==19.10b0" "isort==4.3.21" "mypy==0.782"
```

### Code development

#### Codebase structure

- [ignite](ignite) - Core library files
    - [engine](ignite/engine) - Module containing core classes like Engine, Events, State.
    - [handlers](ignite/handlers) - Module containing out-of-the-box handlers
    - [metrics](ignite/metrics) - Module containing out-of-the-box metrics
    - [contrib](ignite/contrib) - Contrib module with other metrics, handlers classes that may require additional dependencies
    - [distributed](ignite/distributed) - Module with helpers for distributed computations
- [tests](tests) - Python unit tests
- [examples](examples) - Examples and notebook tutorials
- [docs](docs) - Documentation files

If you modify the code, you will most probably also need to code some tests to ensure the correct behaviour. We are using 
`pytest` to write our tests:
  - naming convention for files `test_*.py`, e.g. `test_precision.py`
  - naming of testing functions `def test_*`, e.g. `def test_precision_on_random_data()`

New code should be compatible with Python 3.X versions. Once you finish implementing a feature or bugfix and tests, 
please run lint checking and tests:

#### Formatting Code

To ensure the codebase complies with a style guide, we use [flake8](https://flake8.pycqa.org/en/latest/),
[black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/) tools to
format and check codebase for compliance with PEP8.
  
##### Formatting without pre-commit

If you choose not to use pre-commit, you can take advantage of IDE extensions configured to black format or invoke 
black manually to format files and commit them.

```bash
# This should autoformat the files
isort -rc .
black .
# Run lint checking
flake8 ignite/ tests/ examples/
# If everything is OK, then commit
git add .
git commit -m "Added awesome feature"
```

#### Formatting with pre-commit

To automate the process, we have configured the repo with [pre-commit hooks](https://pre-commit.com/) to use black to autoformat the staged files to ensure every commit complies with a style guide. This requires some setup, which is described below:

1. Install pre-commit in your python environment.
2. Run pre-commit install that configures a virtual environment to invoke black, isort and flake8 on commits.

```bash
pip install pre-commit
pre-commit install
```

3. When files are committed:
    - If the stages files are not compliant with black, black will autoformat the staged files. If this were to happen, files should be staged and committed again. See example code below.
    - If the staged files are not compliant with flake8, errors will be raised. These errors should be fixed and the files should be committed again. See example code below.
    
```bash
git add .
git commit -m "Added awesome feature"
# DONT'T WORRY IF ERRORS ARE RAISED.
# YOUR CODE IS NOT COMPLIANT WITH flake8, isort or black
# Fix any flake8 errors by following their suggestions
# isort and black will automatically format the files so they might look different, but you'll need to stage the files 
# again for committing
# After fixing any flake8 errors
git add .
git commit -m "Added feature"
```

#### Run tests:

To run a specific test, for example `test_terminate` from `test_engine.py`
```bash
pytest tests/ignite/engine/test_engine.py -vvv -k test_terminate
```
To run all tests with coverage (assuming installed `pytest-cov`):
```bash
CI_PYTHON_VERSION=<your python version, e.g 3.7> bash tests/run_cpu_tests.sh
# for example
# CI_PYTHON_VERSION=3.7 bash tests/run_cpu_tests.sh
```

#### Run Mypy checks:
To run mypy to check the optional static type:
```bash
mypy --config-file mypy.ini
```

To change any config for specif folder, please see the file mypy.ini

#### Send a PR

If everything is OK, please send a Pull Request to https://github.com/pytorch/ignite

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

##### Sync up with the upstream

First, make sure you have set [upstream](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork) by running:

```bash
git remote add upstream https://github.com/pytorch/ignite
```

Then you can see if you have set up multiple remote correctly by running `git remote -v`:

```bash
origin  https://github.com/{YOUR_USERNAME}/ignite.git (fetch)
origin  https://github.com/{YOUR_USERNAME}/ignite.git (push)
upstream        https://github.com/pytorch/ignite (fetch)
upstream        https://github.com/pytorch/ignite (push)
```

Now you can get the latest development into your forked repository with this:

```bash
git fetch upstream
git checkout master
git merge upstream/master
```

### Writing documentation

Ignite uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 120 characters.

#### Local documentation building and deploying

Please, follow the instructions to build and deploy the documentation locally.

##### Install requirements

```bash
cd docs
pip install -r requirements.txt
```

##### Build

```bash
cd docs
make html
```

##### Local deployment

Please, use python 3.X for the command below:
```bash
cd docs/build
python -m http.server <port>
# python -m http.server 1234
```
Then open the browser at `0.0.0.0:<port>` (e.g. `0.0.0.0:1234`) and click to `html` folder.
