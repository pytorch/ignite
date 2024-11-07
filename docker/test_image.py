#
# Tests :
# For all images
#     can import torch and its version == required one
#     can import ignite and its version == required one
# for all -vision images
#     can import opencv without driver issue
# for all horovod images
#     can import horovod and its version == required one
#
import argparse
import importlib
import os


def check_package(package_name, expected_version=None):
    mod = importlib.import_module(package_name)

    if expected_version is not None:
        assert hasattr(mod, "__version__"), f"Imported package {package_name} does not have __version__ attribute"
        version = mod.__version__
        # Remove all +something from the version name: e.g torch 2.5.1+cu124
        if "+" in version:
            old_version = version
            version = version.split("+")[0]
            print(f"Transformed version: {old_version} -> {version}")
        assert (
            version == expected_version
        ), f"Version mismatch for package {package_name}: got {version} but expected {expected_version}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Check docker image script")
    parser.add_argument("image", type=str, help="Docker image to check")
    args = parser.parse_args()

    docker_image_name = args.image
    name, version = docker_image_name.split(":")
    assert version != "latest", version
    torch_version, ignite_version = version.split("-")
    _, image_type = name.split("/")

    check_package("torch", expected_version=torch_version)
    check_package("ignite", expected_version=ignite_version)

    if "hvd" in image_type:
        assert "HVD_VERSION" in os.environ
        val = os.environ["HVD_VERSION"]
        hvd_version = val if val[0] != "v" else val[1:]
        check_package("horovod", expected_version=hvd_version)

    if "msdp" in image_type:
        assert "MSDP_VERSION" in os.environ
        val = os.environ["MSDP_VERSION"]
        hvd_version = val if val[0] != "v" else val[1:]
        check_package("deepspeed", expected_version=hvd_version)

    if "vision" in image_type:
        check_package("cv2")

    if "nlp" in image_type:
        check_package("transformers")

    if "apex" in image_type:
        check_package("apex")
