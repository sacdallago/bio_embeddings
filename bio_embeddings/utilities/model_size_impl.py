"""Due to multiprocessing constraints, this needs to be a separate file"""
import subprocess
from typing import Any, Union

import torch
from deepblast.trainer import LightningAligner

from bio_embeddings.embed import name_to_embedder
from bio_embeddings.extract import BasicAnnotationExtractor
from bio_embeddings.project import PBTucker
from bio_embeddings.utilities import get_model_file


def get_model(name: str, device: Union[None, str, torch.device]) -> Any:
    if name in ["bert_from_publication", "seqvec_from_publication"]:
        return BasicAnnotationExtractor(name, device)
    elif name == "esm1v":
        return name_to_embedder[name](ensemble_id=1, device=device)
    elif name in name_to_embedder:
        return name_to_embedder[name](device=device)
    elif name == "pb_tucker":
        return PBTucker(get_model_file("pb_tucker", "model_file"), device)
    elif name == "deepblast":
        model_file = get_model_file("deepblast", "model_file")
        return LightningAligner.load_from_checkpoint(model_file).to(device)
    else:
        raise ValueError(f"Unknown name {name}")


def get_cpu_size(name: str) -> int:
    """Returns bytes"""
    import os
    import psutil

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    _model = get_model(name, None if name == "unirep" else "cpu")
    process = psutil.Process(os.getpid())
    with_model = process.memory_info().rss
    del _model
    return with_model - baseline


def get_gpu_size(name: str) -> int:
    """Returns MiB"""
    import os

    pid = os.getpid()
    _model = get_model(name, "cuda")
    smi_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader",
    ]
    pseudo_csv = subprocess.check_output(smi_command, text=True)
    del _model

    for line in pseudo_csv.splitlines():
        if line.startswith(str(pid)):
            return int(line.split(", ")[1].replace(" MiB", ""))
    else:
        raise ValueError(f"{pid}\n{pseudo_csv}")
