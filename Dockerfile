FROM nvcr.io/nvidia/pytorch:24.12-py3

ARG TZ="Australia/Adelaide"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND="noninteractive"
# mitigates https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
WORKDIR /workspace
RUN pip install  uv --root-user-action=ignore && uv --version && \
    apt-get update && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y && \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2/
RUN uv pip install --upgrade  pip setuptools wheel --break-system-packages
RUN uv pip install aiohttp --break-system-packages
RUN uv pip install --no-build-isolation opencv-python-headless --break-system-packages
RUN uv pip install --no-build-isolation "comfyui@git+https://github.com/samhodge-aiml/ComfyUI.git@cde73655d220827b6c934ba145037e4f499670df" --break-system-packages
RUN rm -rf /var/lib/apt/lists/*
RUN uv pip install  git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite --break-system-packages
RUN uv pip install comfyui-hunyuanvideowrapper@git+https://github.com/samhodge-aiml/ComfyUI-HunyuanVideoWrapper@0a1e01ca20ff29166953c7dcee18a40503a752ec --break-system-packages 
RUN uv pip install comfyui-custom-scripts@git+https://github.com/samhodge-aiml/ComfyUI-Custom-Scripts@4fbb519c4e9616481ba16b4cd4943dbfabc6d553 --break-system-packages
RUN uv pip install comfyui-kjnodes@git+https://github.com/samhodge-aiml/ComfyUI-KJNodes@31a6e7ecf3d7c954d1eee0a829b37a8c13de7c92 --break-system-packages
RUN uv pip install comfyui-comfyroll@git+https://github.com/samhodge-aiml/ComfyUI_Comfyroll_CustomNodes@555273416791b32fa2e059c0a8831262b9a2361f --break-system-packages
RUN uv pip install comfyui-hunyaunloom@git+https://github.com/samhodge-aiml/ComfyUI-HunyuanLoom/@6b37a746408f1acedb4601e4312424991e89f167 --break-system-packages
RUN uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ella.git  --break-system-packages
RUN uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ipadapter-plus  --break-system-packages
RUN uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-layerdiffuse.git  --break-system-packages
RUN uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-bria-bg-removal.git --break-system-packages
RUN uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation --break-system-packages
RUN uv pip install git+https://github.com/samhodge-aiml/appmana-comfyui-nodes-impact-pack/@0984a9212ccf4405bd155b507affd484552d3084 --break-system-packages
#RUN uv pip install git+https://github.com/AppMAna/appmana-comfyui-nodes-tensorrt --break-system-packages
RUN uv pip install comfyui-advanced-controlnet@git+https://github.com/samhodge-aiml/ComfyUI-Advanced-ControlNet/@7931cfef22cb48f2c857cf8053b33cae634dd4d5 --break-system-packages
RUN uv pip install comfyui-jankhidiffusion@git+https://github.com/samhodge-aiml/comfyui_jankhidiffusion/@49fe48e83b6910a590ce4224cbd234ecc18c83a0 --break-system-packages
RUN uv pip install comfyui-sampler-lcm-alternative@git+https://github.com/samhodge-aiml/ComfyUI-sampler-lcm-alternative/@ea0d7660ab70d3c691249d0abc9241cd1c2f2e9c --break-system-packages
RUN uv pip install comfyui-liveportraitkj@git+https://github.com/samhodge-aiml/ComfyUI-LivePortraitKJ/@403fb03c5a14b36fd645fb9d2afa9c6b605a7617 --break-system-packages
RUN uv pip install rgthree-comfy@git+https://github.com/samhodge-aiml/rgthree-comfy/@879eeb029d87bd8dc1a990ab8844743a7304655b --break-system-packages
RUN uv pip install comfyui-clip-with-break@git+https://github.com/samhodge-aiml/comfyui-clip-with-break/@b25f9200b48867664c7766a99a4e9b4ad6a5b913 --break-system-packages
RUN uv pip install git+https://github.com/samhodge-aiml/appmana-comfyui-nodes-animatediff-evolved/@5b56ad9bfd539746b338204002efce54bf19240c --break-system-packages
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
#RUN comfyui --quick-test-for-ci --cpu --cwd /workspace 
EXPOSE 8188
CMD ["python3", "-m", "comfy.cmd.main", "--listen"]
