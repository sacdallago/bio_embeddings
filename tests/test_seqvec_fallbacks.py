from typing import List, Tuple
from unittest import mock

import numpy

from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

# lengths, cpu, success
LogType = List[Tuple[List[int], bool, bool]]

given_limit = 18
actual_limit = 15


class MockElmoMemory:
    cpu: bool
    log: LogType

    def __init__(self, cuda_device: int, log: LogType):
        self.cpu = cuda_device == -1
        self.log = log

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        lengths = [len(sequence) for sequence in batch]
        if not self.cpu:
            if sum(len(i) for i in batch) > actual_limit:
                self.log.append((lengths, False, False))
                raise RuntimeError(f"Too big for the GPU: {[len(i) for i in batch]}")
            else:
                self.log.append((lengths, False, True))
                return [numpy.zeros((3, length, 1024)) for length in lengths]
        else:
            self.log.append((lengths, True, True))
            return [numpy.zeros((3, length, 1024)) for length in lengths]


def test_fallbacks(caplog):
    """Check that the fallbacks to single sequence processing and/or the CPU are working.

    batch_size is 18, actual GPU limit 15, so that we get a case where a too
    big batch has to be handled

    Procedure:
     * [7, 7, 7] Fails, passes with single sequence processing
     * [7, 8] Passes
     * [20] Fails, fails with single sequence processing, passes on the CPU
    """

    elmo_log: LogType = []
    with mock.patch(
        "bio_embeddings.embed.seqvec_embedder.ElmoEmbedder",
        lambda weight_file, options_file, cuda_device: MockElmoMemory(
            cuda_device, elmo_log
        ),
    ), mock.patch(
        "bio_embeddings.utilities.helpers.torch.cuda.is_available", lambda: True
    ):
        sequences = ["M" * 20, "M" * 8, "M" * 8, "M" * 7, "M" * 7, "M" * 7]
        embeddings_generator = SeqVecEmbedder(
            weights_file="/invalid/path", options_file="/invalid/path", warmup_rounds=0
        ).embed_many(sequences, given_limit)
        list(embeddings_generator)

    # lengths, cpu, success
    assert elmo_log == [
        ([20], False, False),
        ([20], True, True),
        ([8, 8], False, False),
        ([8], False, True),
        ([8], False, True),
        ([7, 7], False, True),
        ([7], False, True),
    ]

    assert caplog.messages == [
        "A sequence is 20 residues long, which is longer than your `batch_size` "
        "parameter which is 18",
        "RuntimeError for sequence with 20 residues: Too big for the GPU: [20]. This "
        "most likely means that you don't have enough GPU RAM to embed a protein this "
        "long. Embedding on the CPU instead, which is very slow",
        "Loading model for CPU into RAM. Embedding on the CPU is very slow and you "
        "should avoid it.",
        "Error processing batch of 2 sequences: Too big for the GPU: [8, 8]. You "
        "might want to consider adjusting the `batch_size` parameter. Will try to "
        "embed each sequence in the set individually on the GPU.",
    ]
