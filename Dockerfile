# This location of python in venv-build needs to match the location in the runtime image,
# so we're manually installing the required python environment
FROM ubuntu:20.04 as venv-build

# build-essential is for jsonnet
RUN apt-get update && \
    apt-get install -y curl build-essential python3 python3-pip python3-distutils python3-venv python3-dev python3-virtualenv git && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - --version 1.1.7

COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
WORKDIR /app

RUN python3 -m venv .venv && \
    # Install a recent version of pip, otherwise the installation of manylinux2010 packages will fail
    .venv/bin/pip install -U pip && \
    # Make sure poetry install the metadata for bio_embeddings
    mkdir bio_embeddings && \
    touch bio_embeddings/__init__.py && \
    touch README.md && \
    $HOME/.local/bin/poetry config virtualenvs.in-project true && \
    $HOME/.local/bin/poetry install --no-dev -E all

FROM nvidia/cuda:11.1-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y python3 python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Workaround for when switching the docker user
# https://github.com/numba/numba/issues/4032#issuecomment-547088606
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

COPY --from=venv-build /app/.venv /app/.venv
COPY . /app/

WORKDIR /app

ENTRYPOINT ["/app/.venv/bin/python", "-m", "bio_embeddings.utilities.cli"]

