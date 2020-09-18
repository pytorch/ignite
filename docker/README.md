# Docker for users

We provide Dockerfiles in order to build containerized execution environment that ease the use of Ignite for computer vision and NLP tasks.

These images are also provided with the following Horovod configuration:

```bash
Horovod v0.20.0:

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
    * `docker pull pytorchignite/base:latest`
- [main/Dockerfile.vision](main/Dockerfile.vision): base image with useful computer vision libraries 
    * `docker pull pytorchignite/vision:latest`
- [main/Dockerfile.nlp](main/Dockerfile.nlp): base image with useful NLP libraries 
    * `docker pull pytorchignite/nlp:latest`
- [main/Dockerfile.apex](main/Dockerfile.apex): multi-stage NVIDIA/apex build with latest Pytorch, Ignite image with minimal dependencies
    * `docker pull pytorchignite/apex:latest`
- [main/Dockerfile.apex-vision](main/Dockerfile.nlp): base apex with useful computer vision libraries
    * `docker pull pytorchignite/apex-vision:latest`
- [main/Dockerfile.apex-nlp](main/Dockerfile.nlp): base apex with useful NLP libraries
    * `docker pull pytorchignite/apex-nlp:latest`
- [hvd/Dockerfile.hvd-base](hvd/Dockerfile.hvd-base): multi-stage Horovod build with latest stable PyTorch, Ignite with minimal dependencies
    * `docker pull pytorchignite/hvd-base:latest`
- [hvd/Dockerfile.hvd-vision](hvd/Dockerfile.hvd-vision): base Horovod image with useful computer vision libraries  
    * `docker pull pytorchignite/hvd-vision:latest`
- [hvd/Dockerfile.hvd-nlp](hvd/Dockerfile.hvd-nlp): base Horovod image with useful NLP libraries 
    * `docker pull pytorchignite/hvd-nlp:latest`
- [hvd/Dockerfile.hvd-apex](hvd/Dockerfile.hvd-apex): multi-stage NVIDIA/apex and Horovod build with latest Pytorch, Ignite image with minimal dependencies
    * `docker pull pytorchignite/hvd-apex:latest`
- [hvd/Dockerfile.hvd-apex-vision](hvd/Dockerfile.hvd-apex-vision): base Horovod apex with useful computer vision libraries
    * `docker pull pytorchignite/hvd-apex-vision:latest`
- [hvd/Dockerfile.hvd-apex-nlp](hvd/Dockerfile.hvd-apex-nlp): base Horovod apex with useful NLP libraries
    * `docker pull pytorchignite/hvd-apex-nlp:latest`
- [msdp/Dockerfile.msdp-apex-base](msdp/Dockerfile.msdp-apex-base): multi-stage MSDeepSpeed build with latest Pytorch, Ignite image with minimal dependencies
    * `docker pull pytorchignite/msdp-base:latest`
- [msdp/Dockerfile.msdp-apex-vision](msdp/Dockerfile.msdp-apex-vision): base MSDeepSpeed build with useful computer vision libraries
    * `docker pull pytorchignite/msdp-vision:latest`
- [msdp/Dockerfile.msdp-apex-nlp](msdp/Dockerfile.msdp-apex-nlp): base MSDeepSpeed build with useful NLP libraries
    * `docker pull pytorchignite/msdp-nlp:latest`
    
## How to use

```bash
docker run -it -v $PWD:/workspace/project --network=host --shm-size 16G pytorchignite/base:latest
```

## Building the image yourself

Dockerfiles are supplied to build images with dependencies that ease the use of Ignite for computer vision / NLP tasks:

```bash
cd main
docker build -t pytorchignite/base:latest -f Dockerfile.base . 
```
