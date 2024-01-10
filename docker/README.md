# Docker for users

We provide Dockerfiles in order to build containerized execution environment that ease the use of Ignite for computer vision and NLP tasks.

These images are also provided with the following Horovod configuration:

```bash
Available Frameworks:
    [ ] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [ ] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [ ] MPI
    [X] Gloo
```

## Installation

- [main/Dockerfile.base](main/Dockerfile.base): latest stable PyTorch, Ignite with minimal dependencies
  - `docker pull pytorchignite/base:latest`
- [main/Dockerfile.vision](main/Dockerfile.vision): updated description
  - `docker pull pytorchignite/vision:update-command`
- [main/Dockerfile.nlp](main/Dockerfile.nlp): updated description
  - `docker pull pytorchignite/nlp:update-command`
- [main/Dockerfile.apex](main/Dockerfile.apex): multi-stage NVIDIA/apex build with latest Pytorch, Ignite image with minimal dependencies
  - `docker pull pytorchignite/apex:update-command`
- [main/Dockerfile.apex-vision](main/Dockerfile.nlp): updated description
  - `docker pull pytorchignite/apex-vision:update-command`
- [main/Dockerfile.apex-nlp](main/Dockerfile.nlp): updated description
  - `docker pull pytorchignite/apex-nlp:update-command`
- [hvd/Dockerfile.hvd-base](hvd/Dockerfile.hvd-base): updated description
  - `docker pull pytorchignite/hvd-base:update-command`
- [hvd/Dockerfile.hvd-vision](hvd/Dockerfile.hvd-vision): updated description
  - `docker pull pytorchignite/hvd-vision:update-command`
- [hvd/Dockerfile.hvd-nlp](hvd/Dockerfile.hvd-nlp): updated description
  - `docker pull pytorchignite/hvd-nlp:update-command`
- [hvd/Dockerfile.hvd-apex](hvd/Dockerfile.hvd-apex): multi-stage NVIDIA/apex and Horovod build with latest Pytorch, Ignite image with minimal dependencies
  - `docker pull pytorchignite/hvd-apex:update-command`
  - `docker pull pytorchignite/hvd-apex:update-command`
- [hvd/Dockerfile.hvd-apex-vision](hvd/Dockerfile.hvd-apex-vision): updated description
  - `docker pull pytorchignite/hvd-apex-vision:update-command`
- [hvd/Dockerfile.hvd-apex-nlp](hvd/Dockerfile.hvd-apex-nlp): updated description
  - `docker pull pytorchignite/hvd-apex-nlp:update-command`



## How to use

```bash
docker run -it -v $PWD:/workspace/project --network=host --ipc=host pytorchignite/base:latest /bin/bash
```

## Building the image yourself

## How to use the updated Docker images

Dockerfiles are supplied to build images with dependencies that ease the use of Ignite for computer vision / NLP tasks:

```bash
cd main
# Replace 'update-command' with the appropriate pull command for the updated Docker image
# Example: docker build -t pytorchignite/base:latest -f Dockerfile.base .
```

```bash
cd main
docker build -t pytorchignite/base:latest -f Dockerfile.base .
```
