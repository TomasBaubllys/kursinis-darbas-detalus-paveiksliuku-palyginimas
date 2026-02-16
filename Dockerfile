FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workdir

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/CSer-Tang-hao/BSFA-FSFG.git .

RUN pip install --no-cache-dir \
    torchvision==0.17.2 \
    scikit-image==0.18.1 \
    tqdm \
    matplotlib

RUN mkdir -p data

CMD ["/bin/bash"]
