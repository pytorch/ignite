# Docker for ignite

Build command line :
```
docker build -t ignite:0.4rc.0-py3.6-torch1.4 .
```

Run tests :
```
docker run --rm -it ignite:0.4rc.0-py3.6-torch1.4 pytest tests -vvv
```

Arguments :
```
--build-arg BASE_IMAGE=<image>        # default: nvidia/cuda:10.1-base
--build-arg PYTHON_VERSION=<version>  # default: 3.6
--build-arg PYTORCH_VERSION=<version> # default: 1.4
```