import os
from collections import defaultdict

import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import AP

area_rngs = {
    "all": [0, float("inf")],
    "small": [0, 1024],
    "medium": [1024, 9216],
    "large": [9216, float("inf")],
}
max_det = 100

iou_thrs = torch.tensor([0.5 + (i / 20) for i in range(10)])
rec_thrs = torch.tensor([i / 100 for i in range(101)])


def get_coco_results():
    dir_path = os.getcwd()
    cocoGt = COCO(dir_path + "/gt.json")
    cocoDt = cocoGt.loadRes(dir_path + "/dt.json")

    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval._prepare()
    cocoEval.evaluate()
    cocoEval.accumulate()

    img_gts = defaultdict(lambda: torch.zeros(0, 9))
    img_dts = defaultdict(lambda: torch.zeros(0, 9))

    for i in cocoEval._gts:
        for img in cocoEval._gts[i]:
            id = img["id"]
            img_id = img["image_id"]
            class_id = img["category_id"]
            area = img["area"]
            crowd = img["iscrowd"]
            ignore = img["ignore"]
            xmin, ymin, xmax, ymax = img["bbox"]
            img_gts[img_id] = torch.vstack(
                [img_gts[img_id], torch.tensor([id, class_id, xmin, ymin, xmax, ymax, area, crowd, ignore])]
            )

    for i in cocoEval._dts:
        for img in cocoEval._dts[i]:
            id = img["id"]
            img_id = img["image_id"]
            class_id = img["category_id"]
            area = img["area"]
            confidence = img["score"]
            xmin, ymin, xmax, ymax = img["bbox"]
            img_dts[img_id] = torch.vstack(
                [img_dts[img_id], torch.tensor([id, class_id, xmin, ymin, xmax, ymax, area, crowd, confidence])]
            )

    img_list = set([i for i in img_gts])
    for i in img_dts:
        img_list.add(i)

    return cocoEval, list(img_list), img_gts, img_dts


cocoEval, img_list, img_gts, img_dts = get_coco_results()


def test_wrong_inputs():
    with pytest.raises(ValueError, match="area_rngs should be a dictionary with key as area name"):
        mAP = AP(area_rngs={1: 2}, max_det=max_det, iou_thrs=iou_thrs, rec_thrs=rec_thrs)
    with pytest.raises(ValueError, match="max_det should be a positive integer, got"):
        mAP = AP(area_rngs=area_rngs, max_det=-1, iou_thrs=iou_thrs, rec_thrs=rec_thrs)
    with pytest.raises(ValueError, match="iou_thrs should be a float tensor, got"):
        mAP = AP(area_rngs=area_rngs, max_det=max_det, iou_thrs=torch.zeros(10, dtype=torch.int64), rec_thrs=rec_thrs)
    with pytest.raises(ValueError, match="rec_thrs should be a float tensor, got"):
        mAP = AP(area_rngs=area_rngs, max_det=max_det, iou_thrs=iou_thrs, rec_thrs=torch.zeros(10, dtype=torch.int64))

    mAP = AP(area_rngs=area_rngs, max_det=max_det, iou_thrs=iou_thrs, rec_thrs=rec_thrs)

    with pytest.raises(ValueError, match="Update Data must be of the form"):
        mAP.update(torch.zeros(1))

    with pytest.raises(ValueError, match="detections_tensor should be of size"):
        mAP.update([torch.zeros(1), []])

    with pytest.raises(ValueError, match="ground_truths should be of size"):
        mAP.update([torch.zeros(1, 9), torch.zeros(2, 10)])


def test_against_coco_map():
    mAP = AP(area_rngs=area_rngs, max_det=max_det, iou_thrs=iou_thrs, rec_thrs=rec_thrs)

    for i in img_list:
        mAP.update([img_dts[i], img_gts[i]])

    results = mAP.compute()
    cocoEval.summarize()
    coco_results = cocoEval.stats[:6]
    assert coco_results[0] == pytest.approx(results[0], 1e-2)
    assert coco_results[1] == pytest.approx(results[1], 1e-2)
    assert coco_results[2] == pytest.approx(results[2], 1e-2)
    assert coco_results[3] == pytest.approx(results[3], 1e-2)
    assert coco_results[4] == pytest.approx(results[4], 1e-2)
    assert coco_results[5] == pytest.approx(results[5], 1e-2)


def _test_distrib_integration(device):
    def _test():
        torch.manual_seed(12)

        def update(_, img_id):
            return [img_dts[img_id], img_gts[img_id]]

        engine = Engine(update)

        mAP = AP(area_rngs=area_rngs, max_det=max_det, iou_thrs=iou_thrs, rec_thrs=rec_thrs, device=device)
        mAP.attach(engine, "mAP")

        data = img_list
        engine.run(data=data, max_epochs=1)

        assert "mAP" in engine.state.metrics

        results = engine.state.metrics["mAP"]
        cocoEval.summarize()
        coco_results = cocoEval.stats[:6]
        assert coco_results[0] == pytest.approx(results[0], 1e-2)
        assert coco_results[1] == pytest.approx(results[1], 1e-2)
        assert coco_results[2] == pytest.approx(results[2], 1e-2)
        assert coco_results[3] == pytest.approx(results[3], 1e-2)
        assert coco_results[4] == pytest.approx(results[4], 1e-2)
        assert coco_results[5] == pytest.approx(results[5], 1e-2)

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test()


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


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


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
