# remove pkg-resources as it causes failure when installing https://github.com/pytorch-ignite/sphinxcontrib-versioning
pip uninstall -y pkg-resources setuptools && pip install --upgrade setuptools pip wheel
pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html -U
pip install -r requirements-dev.txt
pip install -r docs/requirements.txt
pip install git+https://github.com/pytorch-ignite/sphinxcontrib-versioning.git
