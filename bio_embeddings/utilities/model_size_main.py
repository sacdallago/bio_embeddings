"""Measure and report the sizes of all models on either CPU or GPU.

python -m bio_embeddings.utilities.model_size_main cpu
python -m bio_embeddings.utilities.model_size_main gpu
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import torch
from tqdm import tqdm

from bio_embeddings.utilities.model_size_impl import get_gpu_size, get_cpu_size


def main():
    parser = ArgumentParser()
    parser.add_argument("device", choices=["cpu", "gpu"])
    args = parser.parse_args()
    device = args.device

    get_size: Callable[[str], int]
    if device == "cpu":
        get_size = get_cpu_size
    else:
        get_size = get_gpu_size

    results = dict()
    other_models = [
        "bert_from_publication",
        "deepblast",
        "pb_tucker",
        "seqvec_from_publication",
    ]
    # all_models = list(name_to_embedder) + other_models
    all_models = other_models
    for name in tqdm(list(sorted(all_models))):
        if name == "unirep" and device == "gpu":
            continue
        with ProcessPoolExecutor() as isolation:
            mib = isolation.submit(get_size, name).result()
            print(name, mib)
            results[name] = mib

    for name, value in results.items():
        print(f"{name} {value:.1f}")
    print()

    if device == "cpu":
        for name, b in results.items():
            gb = b / (1000 ** 3)
            print(f"{name} {gb:.1f}")
    else:
        for name, mib in results.items():
            gb = mib * (1024 ** 2) / (1000 ** 3)
            print(f"{name} {gb:.1f}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # :(

    main()
