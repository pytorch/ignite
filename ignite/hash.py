from argparse import ArgumentParser as __ArgumentParser

from ignite.utils import hash_checkpoint as __hash_checkpoint

if __name__ == "__main__":
    __description = """
Hash the checkpoint file in the format of <filename>-<hash>.<ext>
to be used with `check_hash` of `torch.hub.load_state_dict_from_url`.
"""
    __parser = __ArgumentParser(
        "ignite.hash", "$ python -m ignite.hash <path-to-checkpoint-file> <output-dir>", __description
    )
    __parser.add_argument("checkpoint_path", nargs=1, type=str, help="Path to the checkpoint file.")
    __parser.add_argument(nargs=1, type=str, help="Output directory to store the hashed checkpoint file.")

    __args = __parser.parse_args()
    __hash_checkpoint(__args.checkpoint_path, __args.output)
