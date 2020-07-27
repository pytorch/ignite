FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

#install git
RUN apt-get update && apt-get install -y --no-install-recommends \
         git && \
     rm -rf /var/lib/apt/lists/*

#setup opencv
RUN echo '#!/bin/bash\n\
python -c "import cv2"\n\
res=$?\n\
if [ "$res" -eq "1" ]; then\n\
    echo "Install libglib2.0 libsm6 libxext6 libxrender-dev for opencv"\n\
    apt-get update\n\
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime\n\
    apt-get install -y tzdata\n\
    dpkg-reconfigure --frontend noninteractive tzdata\n\
    apt-get -y install --no-install-recommends libglib2.0 libsm6 libxext6 libxrender-dev\n\
fi\n'\
>> setup_opencv.sh

RUN sh setup_opencv.sh

#setup apex
RUN echo '#!/bin/bash\n\
tmp_apex_path="/tmp/apex"\n\
python -c "import apex"\n\
res=$?\n\
if [ "$res" -eq "1" ]; then\n\
    echo "Setup NVIDIA Apex"\n\
    rm -rf $tmp_apex_path\n\
    git clone https://github.com/NVIDIA/apex $tmp_apex_path\n\
    cd $tmp_apex_path\n\
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"\n\
    pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .\n\
fi\n'\
>> setup_apex.sh

RUN sh setup_apex.sh

RUN echo 'albumentations\n\
          image-dataset-viz\n\
          numpy\n\
          opencv-python\n\
          py_config_runner\n\
          pytorch-ignite\n\
          pillow\n\
          tensorboard\n\
          tqdm\n\
          trains>=0.15.0\n'\
>> requirements.txt

RUN pip install --upgrade --no-cache-dir -r requirements.txt

#create a non-privileged docker user (for debian)
RUN groupadd -g 61000 ignite \
    && useradd -g 61000 -l -M \
        -s /sbin/nologin -u 61000 ignite

RUN chown -R ignite:ignite /workspace

USER ignite

ENTRYPOINT ["/bin/sh"]