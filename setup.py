import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bio_embeddings",
    version="0.0.1",
    author="Christian Dallago",
    author_email="christian.dallago@tum.de",
    description="A package to generate or retrieve NLP embeddings for bioinformatics applications",
    long_description=long_description,
    install_requires=["torch", "allennlp", "numpy", "gensim", "biopython", "ruamel_yaml", "pandas", "h5py", "transformers", "plotly"],
    scripts=['bio_embeddings/utilities/bio_embeddings'],
    long_description_content_type="text/markdown",
    url="https://github.com/sacdallago/bio_embeddings",
    packages=setuptools.find_packages(exclude=["notebooks", "webserver", "examples"]),
    package_data={'': ['*.yml']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)