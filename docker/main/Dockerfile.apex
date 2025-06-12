# Multi-stage build
# Dockerfile.apex

ARG PTH_VERSION

# 1/Building apex with pytorch:*-devel
FROM pytorch/pytorch:${PTH_VERSION}-devel AS apex-builder

ENV CUDA_HOME=/usr/local/cuda

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends git && \
     rm -rf /var/lib/apt/lists/*

# Build apex
RUN echo "Setup NVIDIA Apex" && \
    tmp_apex_path="/tmp/apex" && \
    rm -rf $tmp_apex_path && \
    git clone https://github.com/NVIDIA/apex $tmp_apex_path && \
    cd $tmp_apex_path && \
    pip install packaging && \
    pip wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .

# 2/ Build the runtime image
FROM pytorch/pytorch:${PTH_VERSION}-runtime

COPY --from=apex-builder /tmp/apex/apex-*.whl /tmp/apex/
RUN pip install --no-cache-dir /tmp/apex/apex-*.whl && \
    rm -fr /tmp/apex

# Install tzdata / git
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    apt-get -y install --no-install-recommends tzdata git && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Ignite main dependencies
RUN pip install --upgrade --no-cache-dir pytorch-ignite \
                                         tensorboard \
                                         tqdm \
                                         fire

# replace pillow with pillow-simd
RUN apt-get update && apt-get -y install --no-install-recommends g++ && \
    pip uninstall -y pillow && \
    CC="cc -mavx2" pip install --upgrade --no-cache-dir --force-reinstall pillow-simd && \
    apt-get remove -y g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Checkout Ignite examples only
RUN mkdir -p pytorch-ignite-examples && \
    cd pytorch-ignite-examples && \
    git init && \
    git config core.sparsecheckout true && \
    echo examples >> .git/info/sparse-checkout && \
    git remote add -f origin https://github.com/pytorch/ignite.git && \
    git pull origin master && \
    # rm very large .git folder
    rm -rf .git
