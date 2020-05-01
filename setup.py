import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bio_embeddings",
    version="0.1.0",
    author="Christian Dallago",
    author_email="christian.dallago@tum.de",
    description="A pipeline for protein embedding generation and visualization",
    long_description=long_description,
    install_requires=["torch", "allennlp", "numpy", "gensim", "biopython", "ruamel_yaml", "pandas", "h5py", "transformers", "plotly", "umap-learn", "matplotlib", "scikit-learn", "scipy", "tqdm"],
    scripts=['bio_embeddings/utilities/bio_embeddings.py'],
    long_description_content_type="text/markdown",
    url="https://github.com/sacdallago/bio_embeddings",
    packages=setuptools.find_packages(exclude=["notebooks", "webserver", "examples"]),
    package_data={'': ['*.yml', '*.toml', '*.txt', '*.md']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9 ",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
)