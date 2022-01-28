"""
Runs all notebooks, skipping the pip install part. Run as

```
python -m test-data.test_notebooks
```
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm


def remove_pip_install_from_notebook(notebook):
    # Remove pip install calls because we want to test the local version
    cells = []
    for cell in notebook["cells"]:
        if "pip3 install" in "".join(cell["source"]) or "pip install" in "".join(
            cell["source"]
        ):
            continue
        cells.append(cell)
    notebook["cells"] = cells


def run_notebook(file: Path):
    notebook = json.loads(file.read_text())
    remove_pip_install_from_notebook(notebook)
    nb = nbformat.reads(json.dumps(notebook), as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with TemporaryDirectory() as temp_dir:
        ep.preprocess(nb, {"metadata": {"path": temp_dir}})


def main():
    parser = ArgumentParser()
    parser.add_argument("file", nargs="*")
    args = parser.parse_args()
    if args.file:
        notebooks = [Path(file) for file in args.file]
    else:
        notebooks = list(Path("notebooks").glob("*.ipynb"))
    for file in tqdm(notebooks):
        print(f"Running {file}")
        try:
            run_notebook(file)
        except Exception as e:
            print(f"{file} failed: {e}")
            continue
        else:
            print(f"{file} passed")


if __name__ == "__main__":
    main()
