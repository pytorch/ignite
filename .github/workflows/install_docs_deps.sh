pip install -U pip wheel
pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html -U
pip install -r ../requirements-dev.txt
pip install -r ../docs/requirements.txt
pip install git+https://github.com/vfdev-5/sphinxcontrib-versioning.git
