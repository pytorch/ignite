#multi-stage build
#1/ building apex with pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel AS apex-builder

ARG ARG_TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
ENV TORCH_CUDA_ARCH_LIST=$ARG_TORCH_CUDA_ARCH_LIST

#install git
RUN apt-get update && apt-get install -y --no-install-recommends \
         git && \
     rm -rf /var/lib/apt/lists/*

#building apex
RUN echo "Setup NVIDIA Apex" && \
    tmp_apex_path="/tmp/apex" && \
    rm -rf $tmp_apex_path && \
    git clone https://github.com/NVIDIA/apex $tmp_apex_path && \
    cd $tmp_apex_path && \
    pip wheel --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

#2/ build the runtime image
FROM pytorch/pytorch:latest

COPY --from=apex-builder /tmp/apex/apex-0.1-cp37-cp37m-linux_x86_64.whl apex-0.1-cp37-cp37m-linux_x86_64.whl
RUN pip install --no-cache-dir apex-0.1-cp37-cp37m-linux_x86_64.whl && \
    rm apex-0.1-cp37-cp37m-linux_x86_64.whl

#install tzdata / opencv dependencies / git
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get -y install --no-install-recommends libglib2.0 \
                                               libsm6 \
                                               libxext6 \
                                               libxrender-dev \
                                               git && \
    rm -rf /var/lib/apt/lists/*

#install ignite vison dependencies
RUN pip install --upgrade --no-cache-dir albumentations \
                                         image-dataset-viz \
                                         numpy \
                                         opencv-python \
                                         py_config_runner \
                                         pytorch-ignite \
                                         pillow \
                                         tensorboard \
                                         tqdm \
                                         "trains>=0.15.0"

#create a non-privileged docker user (for debian)
RUN groupadd -g 61000 ignite \
    && useradd -g 61000 -l -M \
        -s /sbin/nologin -u 61000 ignite

RUN chown -R ignite:ignite /workspace

USER ignite

ENTRYPOINT ["/bin/bash"]