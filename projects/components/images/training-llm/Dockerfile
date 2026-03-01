ARG BASE_IMAGE=ray-worker
FROM ${BASE_IMAGE}

# Install LLM specific libraries
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    peft==0.7.1 \
    datasets==2.16.1 \
    bitsandbytes==0.41.3 \
    scipy \
    einops \
    sentencepiece \
    protobuf

# Install DeepSpeed (pre-compiled wheel is faster, but pip install usually compiles JIT)
# For production, we might want to compile this in a multi-stage build or use a pre-built wheel
RUN DS_BUILD_OPS=1 pip install --no-cache-dir deepspeed==0.12.6

# Install Flytekitplugins-ray for using this image in Flyte tasks
RUN pip install --no-cache-dir flytekitplugins-ray==1.10.0

WORKDIR /root
