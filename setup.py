import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read("README.md").replace(
    'src="assets/', 'src="https://raw.githubusercontent.com/pytorch/ignite/master/assets/'
)

VERSION = find_version("ignite", "__init__.py")

requirements = [
    "torch>=1.3,<2",
]

setup(
    # Metadata
    name="pytorch-ignite",
    version=VERSION,
    author="PyTorch Core Team",
    author_email="soumith@pytorch.org",
    url="https://github.com/pytorch/ignite",
    description="A lightweight library to help with training neural networks in PyTorch.",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="BSD",
    # Package info
    packages=find_packages(exclude=("tests", "tests.*",)),
    zip_safe=True,
    install_requires=requirements,
)
