# Dockerfile.base
ARG PTH_VERSION

FROM pytorch/pytorch:${PTH_VERSION}-runtime

# Temporary hack to install pth 1.7.1
RUN conda install -y pytorch torchvision cudatoolkit=11.0 -c pytorch

# Install tzdata / git / g++
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get -y install --no-install-recommends git g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Ignite main dependencies
RUN pip install --upgrade --no-cache-dir pytorch-ignite \
                                         tensorboard \
                                         tqdm

# replace pillow with pillow-simd
RUN pip uninstall -y pillow && \
    CC="cc -mavx2" pip install --upgrade --no-cache-dir --force-reinstall pillow-simd && \
    apt-get autoremove -y

# Checkout Ignite examples only
RUN mkdir -p pytorch-ignite-examples && \
    cd pytorch-ignite-examples && \
    git init && \
    git config core.sparsecheckout true && \
    echo examples >> .git/info/sparse-checkout && \
    git remote add -f origin https://github.com/pytorch/ignite.git && \
    git pull origin master
