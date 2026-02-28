FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workdir

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/TomasBaubllys/kursinis-darbas-detalus-paveiksliuku-palyginimas.git .

WORKDIR /workdir/src

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -U typing-extensions

RUN python download_data.py

CMD ["/bin/bash"]
