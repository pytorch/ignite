# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -j auto -WT --keep-going --color
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ignite
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docset: html
	doc2dash --name $(SPHINXPROJ) --icon $(SOURCEDIR)/_static/img/pytorch-logo-flame.png --enable-js --online-redirect-url https://pytorch.org/ignite/ --force $(BUILDDIR)/html/

	# Manually fix because Zeal doesn't deal well with `icon.png`-only at 2x resolution.
	cp $(SPHINXPROJ).docset/icon.png $(SPHINXPROJ).docset/icon@2x.png
	convert $(SPHINXPROJ).docset/icon@2x.png -resize 16x16 $(SPHINXPROJ).docset/icon.png

rebuild:
	rm -rf source/generated && make clean && make html

clean:
	@echo "Cleaning up..."
	python -c "import shutil; shutil.rmtree('$(BUILDDIR)', ignore_errors=True)"
	python -c "import shutil; shutil.rmtree('$(SOURCEDIR)/generated', ignore_errors=True)"
	python -c "import os; [os.remove(f) for f in os.listdir('.') if f.endswith('.pyc')]"
	python -c "import shutil; import os; [shutil.rmtree(f) for f in os.listdir('.') if f == '__pycache__' and os.path.isdir(f)]"

.PHONY: help Makefile docset

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
