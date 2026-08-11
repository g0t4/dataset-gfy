FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ripgrep fd-find findutils diffutils sed grep coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd

RUN pip install --no-cache-dir jinja2 rich

WORKDIR /workspace
