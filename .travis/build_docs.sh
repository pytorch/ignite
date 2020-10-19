#!/bin/sh
sphinx-versioning --use-master-conf --use-master-templates build --greatest-tag --whitelist-branches master docs/source docs/build/html
# Create .nojekyll file to serve correctly _static and friends
touch docs/build/html/.nojekyll
