#
# Tests :
# For all images
#     can import torch and its version == required one
#     can import ignite and its version == required one
# for all -vision images
#     can import opencv without driver issue
# for all horovod images
#     can import horovod and its version == required one
# for all msdp images
#     can import deepspeed and its version == required one
#
# Requirements:
#   pip install docker
#
import argparse
import docker
import json
import os


def run_python_cmd(cmd):
    try:
        out = client.containers.run(
            args.image, 
            f"python -c '{cmd}'",
            auto_remove=True,
            stderr=True,
        )
        assert isinstance(out, bytes), type(out)
        out = out.decode("utf-8").strip()
    except docker.errors.ContainerError as e:
        raise RuntimeError(e)
    return out


base_cmd = """
def main():
    import torch
    import ignite

    result = dict()
    result["torch"] = torch.__version__
    result["ignite"] = ignite.__version__

    {hvd}
    {msdp}

    print(result)


try:
    main()
except Exception as e:
    print(e)
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Check docker image script")
    parser.add_argument("image", type=str, help="Docker image to check")
    args = parser.parse_args()
    client = docker.from_env()

    docker_image_name = args.image
    name, version = docker_image_name.split(":")
    assert version != "latest", version
    torch_version, ignite_version = version.split("-")
    _, image_type = name.split("/")

    expected_out = {
        "torch": torch_version,
        "ignite": ignite_version,
    }
    
    hvd_cmd = ""
    if "hvd" in image_type:
        hvd_cmd = "import horovod; result[\"hvd\"] = horovod.__version__"
        assert "HVD_VERSION" in os.environ
        val = os.environ["HVD_VERSION"]
        expected_out["hvd"] = val if val[0] != "v" else val[1:]


    msdp_cmd = ""
    if "msdp" in image_type:
        msdp_cmd = "import deepspeed; result[\"msdp\"] = deepspeed.__version__"
        assert "MSDP_VERSION" in os.environ
        val = os.environ["MSDP_VERSION"]
        expected_out["msdp"] = val if val[0] != "v" else val[1:]

    cmd = base_cmd.format(hvd=hvd_cmd, msdp=msdp_cmd)
    out = run_python_cmd(cmd)
    try:
        out = out.replace("\'", "\"")
        out = json.loads(out)
    except json.decoder.JSONDecodeError:
        raise RuntimeError(out)

    for k, v in expected_out.items():
        assert k in out, f"{k} not in {out.keys()}"
        assert v in out[k], f"{v} not in {out[k]}"
    
    if "vision" in image_type:
        run_python_cmd("import cv2")

    if "apex" in image_type:
        run_python_cmd("import apex")
