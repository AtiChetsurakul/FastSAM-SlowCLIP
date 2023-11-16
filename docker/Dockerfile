FROM nvcr.io/nvidia/pytorch:23.07-py3
ARG DEBIAN_FRONTEND=noninteractive
# Set up timezone
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

RUN apt-get update && apt install -y git unzip ffmpeg libsm6 libxext6

RUN pip install open_clip_torch

RUN mkdir -p /home/works/
RUN mkdir -p /home/works/clip-app/

RUN cd /home/works/
RUN echo 'เรื่องราวมันเริ่มเมื่อตอนที่พบเครื่องมือเอเลี่ยนประหลาด'
COPY modelLoader.py modelLoader.py 
RUN python modelLoader.py
WORKDIR /home/works/
RUN pip install grpcio
RUN pip install pymilvus[client]
RUN pip install gdown
RUN pip install -r https://raw.githubusercontent.com/CASIA-IVA-Lab/FastSAM/main/requirements.txt

RUN gdown 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv
RUN pip install opencv-python==4.8.0.74

RUN pip install streamlit
# RUN pip install streamlit pymilvus[client] deep-translator 
# # FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# ARG DEBIAN_FRONTEND=noninteractive
# # Set up timezone
# ENV TZ=Asia/Bangkok
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
#     echo $TZ > /etc/timezone

# RUN apt-get update && apt install -y git unzip ffmpeg libsm6 libxext6
# RUN pip install matplotlib onnxruntime onnx
# RUN pip install grpcio

# RUN mkdir -p /home/works/
# RUN mkdir -p /home/works/solider-app/
# RUN mkdir -p /home/works/calDist-app/
# WORKDIR /home/works/

# RUN cd /home/works/

# COPY PIPrequirements.txt requirements.txt
# RUN pip install -r requirements.txt --no-deps
# COPY Req2.txt requirements2.txt
# RUN pip install -r requirements2.txt --no-deps
# RUN pip install gdown

# RUN mkdir -p /home/works/Weight/SOLIDERREID/msmt
# # IF this path is broken please change the path to our minio server
# RUN gdown 1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs
# RUN mv swin_base_msmt17.pth /home/works/Weight/SOLIDERREID/msmt/ 
# RUN cd /home/works/


