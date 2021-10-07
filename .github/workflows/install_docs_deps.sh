# remove pkg-resources as it causes failure when installing https://github.com/pytorch-ignite/sphinxcontrib-versioning
pip uninstall -y pkg-resources setuptools && pip install --upgrade setuptools pip wheel
pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html -U
pip install -r requirements-dev.txt
pip install -r docs/requirements.txt
pip install git+https://github.com/trsvchn/sphinxcontrib-versioning.git@update-sphinx
# patch autosummary
echo "Patching autosummary..."
sphinx_location=$pythonLocation/lib/python`python -c "import sys; print(sys.version_info.major, sys.version_info.minor, sep='.')"`/site-packages
cp docs/autosummary.patch $sphinx_location
cd $sphinx_location
echo "Applying patch..."
# Some checks
git apply --check autosummary.patch
git apply --summary autosummary.patch
# Now apply
git apply autosummary.patch
echo "Done!"
