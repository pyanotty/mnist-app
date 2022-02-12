# FROM kaggle/python-gpu-build
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04-rc

WORKDIR /model_update

RUN pip install mlflow
RUN pip install hydra-core --upgrade
RUN pip install omegaconf

