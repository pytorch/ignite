# Docker for Ignite examples

Usage:

Pulling ignite-examples image:
```
docker pull pytorchignite/ignite-examples:latest
```
Running "/bin/sh" in the container with a mount point on "examples" from Ignite root folder:
```
docker run -v ${PWD}/examples:/workspace/examples -it pytorchignite/ignite-examples:latest
```
In the container, launching MNIST example:
```
cd examples/mnist
python mnist_with_tensorboard.py --log_dir=/tmp/tensorboard_logs
```
With GPU support: 
```
docker run --gpus 1 -v ${PWD}/examples:/workspace/examples -it pytorchignite/ignite-examples:latest
```
