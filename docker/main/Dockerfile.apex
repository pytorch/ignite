# Multi-stage build
# 1/Building apex with pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel AS apex-builder

ARG ARG_TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
ENV TORCH_CUDA_ARCH_LIST=$ARG_TORCH_CUDA_ARCH_LIST

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends git && \
     rm -rf /var/lib/apt/lists/*

# Build apex
RUN echo "Setup NVIDIA Apex" && \
    tmp_apex_path="/tmp/apex" && \
    rm -rf $tmp_apex_path && \
    git clone https://github.com/NVIDIA/apex $tmp_apex_path && \
    cd $tmp_apex_path && \
    pip wheel --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# 2/ Build the runtime image
FROM pytorch/pytorch:latest

COPY --from=apex-builder /tmp/apex/apex-0.1-cp37-cp37m-linux_x86_64.whl apex-0.1-cp37-cp37m-linux_x86_64.whl
RUN pip install --no-cache-dir apex-0.1-cp37-cp37m-linux_x86_64.whl && \
    rm apex-0.1-cp37-cp37m-linux_x86_64.whl

# Install tzdata / git
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get -y install --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Ignite main dependencies
RUN pip install --upgrade --no-cache-dir pytorch-ignite \
                                         tensorboard \
                                         tqdm

# Checkout Ignite examples only
RUN mkdir -p pytorch-ignite-examples && \
    cd pytorch-ignite-examples && \
    git init && \
    git config core.sparsecheckout true && \
    echo examples >> .git/info/sparse-checkout && \
    git remote add -f origin https://github.com/pytorch/ignite.git && \
    git pull origin master

ENTRYPOINT ["/bin/bash"]