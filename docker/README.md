# Docker for users

We provide Dockerfiles in order to build containerized execution environment that ease the use of Ignite for computer vision and NLP tasks.

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

## How to use

```bash
docker run --rm -it -v $PWD:/workspace/project --network=host --shm-size 16G pytorchignite/base:latest
```
