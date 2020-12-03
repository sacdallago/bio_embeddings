import gc
import subprocess
import sys
from argparse import ArgumentParser
from time import sleep

import torch

from bio_embeddings.embed import name_to_embedder


def embedder_round_trip(name: str) -> int:
    """This function should tell us the size of an embedder and not leak memory"""
    memory_allocated_a = torch.cuda.memory_allocated(0)
    all_allocated_a = torch.cuda.memory_stats()["active.all.allocated"]
    # noinspection PyUnusedLocal
    embedder = name_to_embedder[name]()
    embedder.embed_many(["PROTEIN", "SEQWENCE"])
    memory_allocated_b = torch.cuda.memory_allocated(0)
    all_allocated_b = torch.cuda.memory_stats()["active.all.allocated"]
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    memory_allocated_c = torch.cuda.memory_allocated(0)
    all_allocated_c = torch.cuda.memory_stats()["active.all.allocated"]
    print(name, memory_allocated_a, memory_allocated_b, memory_allocated_c)
    print(name, all_allocated_a, all_allocated_b, all_allocated_c)
    return memory_allocated_b - memory_allocated_a


def main():
    parser = ArgumentParser()
    parser.add_argument("embedder", help="Lowercase name of the embedder or all")
    args = parser.parse_args()
    if args.embedder == "all":
        for name in name_to_embedder:
            subprocess.check_call([sys.executable, __file__, name])
    elif args.embedder == "all-keep":
        ram_by_name = {}
        for name in name_to_embedder:
            print(name)
            ram_by_name[name] = embedder_round_trip(name)
        print(ram_by_name)
        sleep(100) # Let me read nvidia-smi
    else:
        embedder_round_trip(args.embedder)


if __name__ == "__main__":
    main()
