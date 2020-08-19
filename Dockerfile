# This location of python in venv-build needs to match the location in the runtime image,
# so we're manually installing the required python environment
FROM ubuntu:18.04 as venv-build

# build-essential is for jsonnet
RUN apt-get update && \
    apt-get install -y curl build-essential python3 python3-distutils python3-venv python3-dev && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
WORKDIR /app

# Install a recent version of pip, otherwise the installation of manylinux2010 packages will fail
RUN python3 -m venv .venv && \
    .venv/bin/pip install -U pip && \
    python3 $HOME/.poetry/bin/poetry config virtualenvs.in-project true && \
    python3 $HOME/.poetry/bin/poetry install --no-dev -E all

FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=venv-build /app/.venv /app/.venv
COPY . /app/

# Workaround for when switching the docker user
# https://github.com/numba/numba/issues/4032#issuecomment-547088606
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR /app

ENTRYPOINT ["/app/.venv/bin/python", "-m", "bio_embeddings.utilities.cli"]

