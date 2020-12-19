# Dockerfile.vision
FROM pytorchignite/base:latest

# Install opencv dependencies
RUN apt-get update && \
    apt-get -y install --no-install-recommends build-essential \
                                               ca-certificates \
                                               libsm6 \
                                               libxext6 \
                                               libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Ignite vision dependencies
RUN pip install --upgrade --no-cache-dir albumentations \
                                         image-dataset-viz \
                                         numpy \
                                         opencv-python \
                                         py_config_runner \
                                         "trains>=0.15.0"

RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
