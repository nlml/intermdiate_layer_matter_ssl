FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
RUN set -ux
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN echo ""
COPY requirements2.txt /tmp/requirements2.txt
RUN pip install --no-cache -r /tmp/requirements2.txt

RUN conda create -n rapids-21.12 -c rapidsai -c nvidia -c conda-forge rapids=21.12 python=3.8 cudatoolkit=11.0 dask-sql


# docker build --rm -t drssl .
# docker run -it --entrypoint bash --gpus all --ipc=host -v /home/liam/intermdiate_layer_matter_ssl/diabetic_retinopathy:/workspace/diabetic_retinopathy -v /ssd1/jama16-retina-replication/data:/workspace/diabetic_retinopathy/data drssl
# cd /workspace/diabetic_retinopathy/ && mkdir -p outs && CUDA_VISIBLE_DEVICES=1 python ssl_mocov2_mse.py --data-path ./data/eyepacs/bin2 --save-path ./outs

python fine_tune_mocov2.py --save-path ./outsFineTune --data-path ./data/eyepacs/bin2 --ckpt-path 

python fine_tune_mocov2.py --ckpt_path epoch\=46-val_loss\=0.820.ckpt --data-path ./data/eyepacs/bin2 --batch-size 128 --save-path outsfinetune --only_ll --mse_btwin moco-mse