FROM pytorch/pytorch:latest

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

#ignite vision dependencies
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