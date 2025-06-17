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
   - Look at the issues labelled as ["help wanted"](https://github.com/pytorch/ignite/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
   - Pick an issue and comment on the task that you want to work on this feature.
   - If you need more context on a particular issue, please ask and we shall provide.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Developing Ignite](#developing-ignite)
  - [Quickstart guide for first-time contributors](#quickstart-guide-for-first-time-contributors)
  - [Installation](#installation)
  - [Code development](#code-development)
    - [Codebase structure](#codebase-structure)
    - [Checking Code Style](#checking-code-style)
      - [Formatting the code](#formatting-the-code)
      - [Formatting the code with a pre-commit hook](#formatting-the-code-with-a-pre-commit-hook)
    - [Run tests](#run-tests)
      - [Run distributed tests only on CPU](#run-distributed-tests-only-on-cpu)
    - [Run Mypy checks](#run-mypy-checks)
    - [Send a PR](#send-a-pr)
      - [Sync up with the upstream](#sync-up-with-the-upstream)
  - [Writing documentation](#writing-documentation)
    - [Local documentation building and deploying](#local-documentation-building-and-deploying)
      - [Install requirements](#install-requirements)
      - [Build](#build)
      - [Local deployment](#local-deployment)

## Developing Ignite

### Quickstart guide for first-time contributors

<summary>

<details>

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your system.
- Create an isolated conda environment for pytorch-ignite:

```bash
conda create -n pytorch-ignite-dev python=3.11
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

1. Make a fork of the repository on the GitHub (see [here](https://github.com/firstcontributions/first-contributions#fork-this-repository) for details).
   As a result, for example your username is `happy-ignite-developer`, then you should be able to see your fork on the GitHub, e.g https://github.com/happy-ignite-developer/ignite.git

2. Clone your fork locally and setup `upstream`. Assuming your username is `happy-ignite-developer`:

```bash
git clone https://github.com/happy-ignite-developer/ignite.git
cd ignite
git remote add upstream https://github.com/pytorch/ignite.git
git remote -v
```

You might see the following output:

```
origin  https://github.com/happy-ignite-developer/ignite.git (fetch)
origin  https://github.com/happy-ignite-developer/ignite.git (push)
upstream        https://github.com/pytorch/ignite (fetch)
upstream        https://github.com/pytorch/ignite (push)
```

3. Sync and install all necessary dependencies:

```bash
git pull upstream master
pip install -e .
pip install -r requirements-dev.txt
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
  - if test function should run on GPU, please **make sure to add `cuda`** in the test name, e.g. `def test_something_on_cuda()`.
    Additionally, we may want to decorate it with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")`.
    For more examples, please see https://github.com/pytorch/ignite/blob/master/tests/ignite/engine/test_create_supervised.py
  - if test function checks distributed configuration, we have to mark the test as `@pytest.mark.distributed` and additional
    conditions depending on the intended checks. For example, please see
    https://github.com/pytorch/ignite/blob/master/tests/ignite/metrics/test_accuracy.py

New code should be compatible with Python 3.X versions. Once you finish implementing a feature or bugfix and tests,
please run lint checking and tests:

#### Checking Code Style

To ensure the codebase complies with the PEP8 style guide, we use [ruff](https://docs.astral.sh/ruff/)
and [black](https://black.readthedocs.io/en/stable/) to lint and format the codebase respectively.

##### Formatting the code

To automate the process, we have configured the repo with [pre-commit](https://pre-commit.com/).

To format files and commit changes:

```bash
# This should lint and autoformat the files
pre-commit -a
# If everything is OK, then commit
git add .
git commit -m "Added awesome feature"
```

##### Formatting the code with a pre-commit hook

To enable the `pre-commit` hooks follow the steps described below:

1. Run `pre-commit install` to configures a virtual environment to invoke linters and formatters on commits.

```bash
pip install pre-commit
pre-commit install
```

2. When files are committed:
   - If the stages files are not compliant, the tools autoformat the staged files. If this were to happen, files should be staged and committed again. See example code below.
   - If the staged files are not compliant errors will be raised. These errors should be fixed and the files should be committed again. See example code below.

```bash
git add .
git commit -m "Added awesome feature"
# DONT'T WORRY IF ERRORS ARE RAISED.
# YOUR CODE IS NOT COMPLIANT WITH flake8, µsort or black
# Fix any flake8 errors by following their suggestions
# µfmt will automatically format the files so they might look different, but you'll need to stage the files
# again for committing
# After fixing any flake8 errors
git add .
git commit -m "Added feature"
```

#### Run tests:

To run a specific test, for example `test_terminate` from `test_engine.py`:

```bash
pytest tests/ignite/engine/test_engine.py -vvv -k test_terminate
```

To run all tests with coverage (assuming installed `pytest-cov` and `pytest-xdist`):

```bash
bash tests/run_cpu_tests.sh
```

On Windows, distributed tests should be skipped

```bash
SKIP_DISTRIB_TESTS=1 bash tests/run_cpu_tests.sh
```

##### Run distributed tests only on CPU

To run distributed tests only (assuming installed `pytest-xdist`):

```bash
export WORLD_SIZE=2
CUDA_VISIBLE_DEVICES="" pytest --dist=each --tx $WORLD_SIZE*popen//python=python tests/ -m distributed -vvv
```

#### Run Mypy checks:

To run mypy to check the optional static type:

```bash
mypy
```

To change any config for specif folder, please see the file mypy.ini

#### Send a PR

If everything is OK, please send a Pull Request to https://github.com/pytorch/ignite from your fork.

If you are not familiar with creating a Pull Request, here are some guides:

- https://github.com/firstcontributions/first-contributions
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

**NOTE : When sending a PR, please kindly check if the changes are required to run in the CI.**

For example, typo changes in `CONTRIBUTING.md`, `README.md` are not required to run in the CI.
So, please add `[skip ci]` in the PR title to save the resources. Ignite has setup several CIs.

- GitHub Actions
- Netlify

So, please add

- `[skip actions]` for the changes which are not required to run on GitHub Actions,
- `[skip netlify]` for the changes which are not required to run on Netlify PR Preview build, or
- `[skip ci]` for the changes which are not required to run on any CI.

**NOTE : Those skip statements are case sensitive, need open bracket `[` and close bracket `]`.
And, Ignite has followed a convention of starting with `skip` word.**

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

Ignite uses [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#type-annotations)
for formatting docstrings, specially from an example of `Google style with Python 3 type annotations` and

- [`.. versionadded::`] directive for adding new classes, class methods, functions,
- [`.. versionchanged::`] directive for adding new arguments, changing internal behaviours, fixing bugs and
- [`.. deprecated::`] directive for deprecations.

Examples: `versionadded` usage [link](https://github.com/pytorch/ignite/blob/52c69251dd9d97c32da1df0477ec3854e5702029/ignite/handlers/state_param_scheduler.py#L24), `versionchanged` usage [link](https://github.com/pytorch/ignite/blob/d2020e4e253ac1455a757c2db895c68ccfd2b958/ignite/metrics/metric.py#L281-L282)

Length of line inside docstrings block must be limited to 120 characters.

[`.. versionadded::`]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded
[`.. versionchanged::`]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionchanged
[`.. deprecated::`]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-deprecated

#### Local documentation building and deploying

Please, follow the instructions to build and deploy the documentation locally.

##### Install requirements

```bash
cd docs
pip install -r requirements.txt
```

[Katex](https://katex.org/) is also needed to build the documentation.
To install katex, you need to have [nodejs](https://nodejs.org/en/) installed.
Optionaly, we can install `nodejs/npm` using conda: `conda install nodejs`.
Then you can install katex with [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/) (if installed).

```bash
npm install -g katex
# or if you use yarn package manager
yarn global add katex
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

Then open the browser at `localhost:<port>` (e.g. `localhost:1234`) and click to `html` folder.

#### Examples testing (doctests)

PyTorch-Ignite uses **Sphinx directives**. Every code that needs to be tested
should be under `.. testcode::` and expected output should be under
`.. testoutput::`. For example:

```py
.. testcode::

    def process_function(engine, batch):
        y_pred, y = batch
        return y_pred, y
    engine = Engine(process_function)
    metric = SSIM(data_range=1.0)
    metric.attach(engine, 'ssim')
    preds = torch.rand([4, 3, 16, 16])
    target = preds * 0.75
    state = engine.run([[preds, target]])
    print(state.metrics['ssim'])

.. testoutput::

    0.9218971...
```

If the floating point results are needed for assertion and the results can vary per operating systems and PyTorch versions, we could assert the results up to 4 or 6 decimal places and match the rest of the results with `...`. Learn more about `sphinx.ext.doctest` in [the official documentation](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html).

To make writing doctests easy, there are some configuratons defined in `conf.py`. Search `doctest_global_setup` in [conf.py](docs/source/conf.py) to see which variables and functions are available.

To run doctests locally:

```sh
cd docs
make html && make doctest
```
