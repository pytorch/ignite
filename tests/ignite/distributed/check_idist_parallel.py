import argparse

import torch

import ignite.distributed as idist


def training(local_rank, config, **kwargs):

    import time

    time.sleep(idist.get_rank() * 0.1)

    print(idist.get_rank(), ": run with config:", config, "- kwargs:", kwargs, f"- backend={idist.backend()}")

    t = torch.tensor([idist.get_rank()], device=idist.device())
    t = idist.all_reduce(t)
    t = t.item()
    ws = idist.get_world_size()
    assert t == ws * (ws - 1) / 2, f"{t} vs {ws}"
    assert local_rank == idist.get_local_rank()


if __name__ == "__main__":
    """
    Usage:

        - No distributed configuration:
        ```
        python tests/ignite/distributed/check_idist_parallel.py
        ```

        - Launch 4 procs using gloo backend with `torch.distributed.launch`
        ```
        python -m torch.distributed.launch --nproc_per_node=4 --use_env \
            tests/ignite/distributed/check_idist_parallel.py --backend=gloo
        ```

        - Launch 2 procs in 2 nodes using gloo backend with `torch.distributed.launch`:
        ```
        bash -c "python -m torch.distributed.launch --nnodes=2 --node_rank=0 \
            --master_addr=localhost --master_port=3344 --nproc_per_node=2 --use_env \
            tests/ignite/distributed/check_idist_parallel.py --backend=gloo &" \
        && bash -c "python -m torch.distributed.launch --nnodes=2 --node_rank=1 \
            --master_addr=localhost --master_port=3344 --nproc_per_node=2 --use_env \
            tests/ignite/distributed/check_idist_parallel.py --backend=gloo &"
        ```

        - Spawn 4 procs in single node using gloo backend:
        ```
        python tests/ignite/distributed/check_idist_parallel.py --backend=gloo --nproc_per_node=4
        ```

        - Spawn 2 procs in 2 nodes using gloo backend:
        ```
        bash -c "python tests/ignite/distributed/check_idist_parallel.py --backend=gloo \
            --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=3344 &" \
        && bash -c "python tests/ignite/distributed/check_idist_parallel.py --backend=gloo \
            --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=localhost --master_port=3344 &"
        ```

        - Spawn 8 procs in single node using xla-tpu backend:
        ```
        python tests/ignite/distributed/check_idist_parallel.py --backend=xla-tpu --nproc_per_node=8
        ```


    """

    parser = argparse.ArgumentParser("Check idist.Parallel")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--nproc_per_node", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--node_rank", type=int, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=str, default=None)

    args = parser.parse_args()

    config = {"model": "resnet18", "lr": 0.01}

    dist_config = dict(
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )

    with idist.Parallel(backend=args.backend, **dist_config) as parallel:
        parallel.run(training, config, a=1, b=2)
