# Basic MNIST Example with Ignite

ported from [pytorch-examples](https://github.com/pytorch/examples/tree/master/mnist)

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`

#### Usage:

Run the example:
```
python mnist.py
```

### Logging with Tensorboard

MNIST example with training and validation monitoring using Tensorboard. Notice 
that if PyTorch version is less than 1.2, the module TensorboardX is required.

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch) (if and only if `PyTorch <= 1.2`): `pip install tensorboardX`
- Tensorboard: `pip install tensorboard`

#### Usage:

Run the example:
```bash
python mnist_with_tensorboard.py --log_dir=/tmp/tensorboard_logs
```

Start tensorboard:
```bash
tensorboard --logdir=/tmp/tensorboard_logs/
```

### Logging with Visdom

MNIST example with training and validation monitoring using Visdom

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [Visdom](https://github.com/facebookresearch/visdom): `pip install visdom`

#### Usage:

Start visdom:
```bash
python -m visdom.server
```

Run the example:
```bash
python mnist_with_visdom.py
```


### Training save & resume

Example shows how to save a checkpoint of the trainer, model, optimizer, lr scheduler. 
User can resume the training from stored latest checkpoint. In addition, training crash can be emulated.

We provided an option `--deterministic` which setups a deterministic trainer as 
[`DeterministicEngine`](https://pytorch.org/ignite/engine.html#ignite.engine.deterministic.DeterministicEngine).
Trainer performs dataflow synchronization on epoch in order to ensure the same dataflow when training is resumed.   
Please, see the documentation for more details.

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- Tensorboard: `pip install tensorboard`

#### Usage:

Training
```bash
python mnist_save_resume_engine.py --log_dir=/tmp/mnist_save_resume/
# or same in deterministic mode
python mnist_save_resume_engine.py --log_dir=/tmp/mnist_save_resume/ --deterministic
```

Resume the training
```bash
python mnist_save_resume_engine.py --log_dir=/tmp/mnist_save_resume/ --resume_from=/tmp/mnist_save_resume/checkpoint_<N>.pt
# or same in deterministic mode
python mnist_save_resume_engine.py --log_dir=/tmp/mnist_save_resume/ --resume_from=/tmp/mnist_save_resume/checkpoint_<N>.pt --deterministic
```

Start tensorboard:
```bash
tensorboard --logdir=/tmp/mnist_save_resume/
```

#### Usage with simulated crash

Initial training with a crash
```bash
python mnist_save_resume_engine.py --crash_iteration 1700 --log_dir=logs --epochs 3
# or same in deterministic mode
python mnist_save_resume_engine.py --crash_iteration 1700 --log_dir=logs --epochs 3 --deterministic
```

Resume from the latest checkpoint
```bash
python mnist_save_resume_engine.py --resume_from logs/checkpoint_1650.pt --log_dir=logs --epochs 3
# or same in deterministic mode
python mnist_save_resume_engine.py --resume_from logs/checkpoint_1650.pt --log_dir=logs --epochs 3 --deterministic
```

The script logs batch stats (mean/std of images, median of targets), model weights norms and computed gradients norms in 
`run.log` and `resume_run.log` to compare training behaviour in both cases. 
If set `--deterministic` option, we can observe the same values after resuming the training.

![tb1](assets/save_resume_p1.png)
![tb2](assets/save_resume_p2.png)