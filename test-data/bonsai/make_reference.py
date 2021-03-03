#!/usr/bin/env python3
import h5py


def main():
    with h5py.File(
        "knn_reference_embedding/knn_reference_embed/reduced_embeddings_file.h5"
    ) as input, h5py.File("knn_reference.h5", "w") as output:
        for key, value in input.items():
            output.create_dataset(value.attrs["original_id"], data=value)


if __name__ == "__main__":
    main()
