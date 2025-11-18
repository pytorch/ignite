# Using DALI with Ignite:

In this example we can see how to build an all-GPU data loading pipeline with DALI and feed it into an Ignite trainer.

DALI does a lot of things that are useful on the data loading and augmentation side - including image decode, transformation, and even multi GPU data pipelining. It is unique in that once DALI applies any GPU operation to a batch, the batch remains on GPU for further transformations. It stays there after the transformations for training until Ignite is done with it - meaning you can save costly CPU-GPU communication.

# Requirements:

- NVIDIA GPU with appropriate drivers
- CUDA 9.0 or later
- PyTorch (make sure it is properly configured to work with your GPU)
- Ignite
- DALI (https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)

### Alternatives:

Google Colabs give users access to free GPU Jupyter environments - the notebook works in those!
https://colab.research.google.com/notebooks/welcome.ipynb

Specifically, there is a Colab friendly version of the notebook here:
https://colab.research.google.com/drive/1F_7DihE8YUzirvWV8xn1aMe0EMAP9iB6

You can also skip a lot of the environment/setup stuff if it is giving you trouble with Docker. There is a PyTorch image that works with this notebook here 
https://ngc.nvidia.com/catalog/containers/nvidia:pytorch

The one liner to get into Jupyter (once nvidia-docker is installed) looks like:
```bash
nvidia-docker run -it --net=host -v $(pwd):/workspace/content nvcr.io/nvia/pytorch:20.01-py3 jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' content/
```

(the 'nvidia-docker' part may be different based on your nvidia-docker version, e.x. "docker run --gpus all" or similar)


# Files:

__Ignite+DALI.ipynb__

The main example notebook. Walks through construction of a DALI pipeline the same way you would Compose() a normal PyTorch transform pipeline, and then feed it into Ignite. Compares loading images from folders using DALI and PyTorch's respective ImageFolder loaders.

__dali_transform_utilities.py__

Some useful DALI utilities that let you `ComposeOps()` DALI operations easily, and then turn it into a DALI graph. Specifically, `ComposeOps`, `TransformPipeline`, `DALILoader`.

__dali_example_utilities.py__

A custom trainer that allows for MovingAverage metrics and prepare_batch logic that allow Ignite to consume DALIIterators. Additionally, add a progressbar and timers for visualization of performance and for fun.

# Thanks:

Great work and many thanks to user chicham, who put together the excellent DALI transformation utilities this example pulls from in Ignite pull 493 (https://github.com/pytorch/ignite/pull/493)!
