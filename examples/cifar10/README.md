# CIFAR10 Example with Ignite

- single GPU
- multi GPU on a single node

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`

#### Usage:

Run the example:
```bash
python main.py
```
## Distributed training

```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="batch_size=256;dist_backend='nccl'"
```


#### Usage:

Train MNIST by using three GPUs in two nodes.

* Single node, multiple GPUs (ex. a node has two GPUs):
  
   Open a terminal and run the example on the GPU 0 (process rank 0):
   ```
   python mnist_dist.py --world_size 2 --rank 0 --gpu 0 --dist_method=file:///tmp/tmp.dat
   ```
   and in another terminal, run the example on the GPU 1 (process rank 1):
   ```
   python mnist_dist.py --world_size 2 --rank 1 --gpu 1 --dist_method=file:///tmp/tmp.dat
   ```

* Multiple nodes with multiple GPUs  (ex. two nodes have two GPUs, respectively):

  On the **Node 0**: 
  - open a terminal and run the example on the GPU 0 (process rank 0):
  ```
  python mnist_dist.py --world_size 4 --rank 0 --gpu 0 --dist_method='tcp://IP_OF_NODE0:FREEPORT'
  ```

  - open another terminal and run the example on the GPU 1 (process rank 1):
  ```
  python mnist_dist.py --world_size 4 --rank 1 --gpu 1 --dist_method='tcp://IP_OF_NODE0:FREEPORT'
  ```

  On the **Node 1**: 
  - open a terminal and  run the example on the GPU 0 (process rank 2):
  ```
  python mnist_dist.py --world_size 4 --rank 2 --gpu 0 --dist_method='tcp://IP_OF_NODE0:FREEPORT'
  ```

  - open another terminal and run the example on the GPU 1 (process rank 3):
  ```
  python mnist_dist.py --world_size 4 --rank 3 --gpu 1 --dist_method='tcp://IP_OF_NODE0:FREEPORT'
  ```

