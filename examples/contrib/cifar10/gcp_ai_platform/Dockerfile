FROM pytorch/pytorch:latest

WORKDIR /workspace

# Setup dependencies
RUN pip install tqdm tensorboardX git+https://github.com/pytorch/ignite.git@distrib

# Setup dataset
RUN mkdir /workspace/data && python -c 'from torchvision.datasets.cifar import CIFAR10; CIFAR10("/workspace/data", download=True)'

# Copy code
RUN mkdir /workspace/code
COPY fastresnet.py /workspace/code/fastresnet.py
COPY main.py /workspace/code/main.py
COPY utils.py /workspace/code/utils.py
COPY gcp_ai_platform/entrypoint.sh /workspace/code/entrypoint.sh
COPY gcp_ai_platform/parse_cluster_spec.py /workspace/code/parse_cluster_spec.py

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["/bin/bash", "/workspace/code/entrypoint.sh"]
