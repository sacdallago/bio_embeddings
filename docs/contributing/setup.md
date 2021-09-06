# Development Setup

bio_embeddings uses [poetry](https://github.com/python-poetry/poetry) to manage dependencies.

* Install [poetry](https://github.com/python-poetry/poetry#installation).
* Run `poetry config virtualenvs.in-project true`. This will make sure all python dependencies will be in a folder called `.venv` (unless you're using conda).
* Clone the repository (`git pull https://github.com/sacdallago/bio_embeddings`)
* Run `poetry install -E all`. This will create a new virtualenv, which you can activate with `poetry shell` or `. .venv/bin/activate` (use `deactivate` to get back to your normal environment). If you're already in a conda environment, poetry will use that environment instead.
* To check that the environment is active, open a python console and run `import bio_embeddings`

## Tests

We use [pytest](https://docs.pytest.org/) to check our code, so can run the tests with `pytest`. Running them all is slow however and takes a lot of disk space, so you can use `SKIP_SLOW_TESTS=1 pytest` to only run a few fast tests.

Some tests that need `RUN_VERY_SLOW_TESTS=1` to be run because they can take a couple of minutes each. FOr example you need `RUN_VERY_SLOW_TESTS=1 pytest tests/conservation.py` to run the test of the conservation predictor because it uses the large T5 language model.

To create a new test, either add a new function in an existing file under `tests/`, or create a new file starting with `test_` in that folder. All functions inside a `test_*.py` file starting with `test_` are run by pytest.

To get the project root as [pathlib.Path](https://docs.python.org/3/library/pathlib.html#basic-use), use `pytestconfig.rootpath`, where pytest will pass `pytestconfig` to your method. Here, we just check the number of entries in `test-data/mapping_file.csv`:

```python
from bio_embeddings.utilities import read_mapping_file


def test_mapping_file_length(pytestconfig):
    mapping_file_path = str(
        pytestconfig.rootpath.joinpath("test-data/mapping_file.csv")
    )
    mapping_file = read_mapping_file(mapping_file_path)
    # Check that the mapping file actually has two rows with data
    assert len(mapping_file) == 2
```

Note that our CI machine doesn't have a GPU, so the tests still need to pass without a GPU. For tests that need a GPU you can use the following:

```python
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Can't test the GPU if there isn't any"
)
def test_my_feature():
    ...
```

Note that in CI, we skip some embedder tests marked `SKIP_NEGLEGTED_EMBEDDER_TESTS` for stale and barely used embedder.
