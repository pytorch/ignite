import os
from collections import defaultdict

import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import MeanAveragePrecision


def create_ground_truths(bbox, category_id=None, image_id=None):
    m = len(bbox)
    if category_id is None:
        category_id = [1] * m
    if image_id is None:
        image_id = [0] * m
    area = [b[2] * b[3] for b in bbox]
    annotations = [
        {
            "area": a,
            "iscrowd": 0,
            "image_id": i,
            "bbox": b,
            "category_id": c,
            "id": k + 1,  # start from 1 !
            "ignore": 0,
            "_ignore": 0,
        }
        for k, (a, i, b, c) in enumerate(zip(area, image_id, bbox, category_id))
    ]
    categories = [{"supercategory": f"_{c}", "id": c, "name": f"{c}"} for c in category_id]
    images = [{"id": i} for i in image_id]
    return {"annotations": annotations, "categories": categories, "images": images}


def create_predictions(bbox, score, category_id=None, image_id=None):
    m = len(bbox)
    if category_id is None:
        category_id = [1] * m
    if image_id is None:
        image_id = [0] * m
    area = [b[2] * b[3] for b in bbox]
    return [
        {
            "image_id": i,
            "category_id": c,
            "bbox": b,
            "score": s,
            "area": a,
            "id": k + 1,  # start from 1 !
            "iscrowd": 0,
        }
        for k, (a, i, b, c, s) in enumerate(zip(area, image_id, bbox, category_id, score))
    ]


def create_tensors(data, keys, device):
    img_tensors = defaultdict(lambda: defaultdict(list))
    for i in data:
        for img in data[i]:
            img_id = img["image_id"]
            for key in keys:
                img_tensors[img_id][key].append(torch.tensor(img[key]))

    for img_id in img_tensors:
        img_tensors[img_id]["image_id"] = img_id
        for key in keys:
            img_tensors[img_id][key] = torch.stack(img_tensors[img_id][key]).to(device)

    return img_tensors


def prepare_coco(device):
    # pycocotools format for bbox
    # (xmin, ymin, width, height)
    bbox = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0, 1.0],
    ]
    # category per bbox (optional)
    category_id = [1, 2]
    # image per bbox (optional)
    image_id = [0, 1]
    # scores per bbox
    score = [0.8, 0.9]

    gt = create_ground_truths(bbox=bbox, image_id=image_id, category_id=category_id)

    pt = create_predictions(bbox=bbox, image_id=image_id, category_id=category_id, score=score)

    coco_gt = COCO()

    coco_gt.dataset = gt
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(pt)

    evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    gt_keys = ["id", "category_id", "area", "iscrowd", "ignore", "bbox"]
    dt_keys = ["id", "category_id", "score", "bbox"]

    img_gts = create_tensors(evaluator._gts, gt_keys, device)
    img_dts = create_tensors(evaluator._dts, dt_keys, device)

    return evaluator, img_gts, img_dts


def test_against_coco_map():
    mAP = MeanAveragePrecision()

    evaluator, img_gts, img_dts = prepare_coco("cpu")

    coco_results = evaluator.stats[:6]

    img_list = set([i for i in img_gts])
    for i in img_dts:
        img_list.add(i)

    for i in img_list:
        mAP.update([[img_dts[i], img_gts[i]]])

    results = mAP.compute()

    assert coco_results[0] == pytest.approx(results["all"], 1e-2)
    assert coco_results[1] == pytest.approx(results["all@0.5"], 1e-2)
    assert coco_results[2] == pytest.approx(results["all@0.75"], 1e-2)
    assert coco_results[3] == pytest.approx(results["small"], 1e-2)
    assert coco_results[4] == pytest.approx(results["medium"], 1e-2)
    assert coco_results[5] == pytest.approx(results["large"], 1e-2)


def _test_distrib_integration(device):
    def _test(metric_device):
        torch.manual_seed(12)

        def update(_, img_id):
            return [[img_dts[img_id], img_gts[img_id]]]

        engine = Engine(update)

        mAP = MeanAveragePrecision(device=metric_device)
        mAP.attach(engine, "mAP")

        evaluator, img_gts, img_dts = prepare_coco(metric_device)

        img_list = set([i for i in img_gts])
        for i in img_dts:
            img_list.add(i)

        data = img_list
        engine.run(data=data, max_epochs=1)

        assert "mAP" in engine.state.metrics

        results = engine.state.metrics["mAP"]
        coco_results = evaluator.stats[:6]
        assert coco_results[0] == pytest.approx(results["all"], 1e-2)
        assert coco_results[1] == pytest.approx(results["all@0.5"], 1e-2)
        assert coco_results[2] == pytest.approx(results["all@0.75"], 1e-2)
        assert coco_results[3] == pytest.approx(results["small"], 1e-2)
        assert coco_results[4] == pytest.approx(results["medium"], 1e-2)
        assert coco_results[5] == pytest.approx(results["large"], 1e-2)

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        _test(metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
