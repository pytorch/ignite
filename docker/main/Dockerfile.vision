# Dockerfile.vision
FROM pytorchignite/base:latest

# Install opencv dependencies
RUN apt-get update && \
    apt-get -y install --no-install-recommends libglib2.0 \
                                               libsm6 \
                                               libxext6 \
                                               libxrender-dev \
                                               libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Ignite vision dependencies
RUN pip install --upgrade --no-cache-dir albumentations \
                                         image-dataset-viz \
                                         numpy \
                                         opencv-python-headless \
                                         py_config_runner \
                                         clearml
