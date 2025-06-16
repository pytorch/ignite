# remove pkg-resources as it causes failure when installing https://github.com/pytorch-ignite/sphinxcontrib-versioning
uv pip uninstall pkg-resources setuptools && pip install --upgrade setuptools pip wheel
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -U
uv pip install -r requirements-dev.txt
uv pip install -r docs/requirements.txt
uv pip install git+https://github.com/pytorch-ignite/sphinxcontrib-versioning.git
