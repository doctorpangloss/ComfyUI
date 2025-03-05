#!/usr/bin/env bash
echo "starting downloads"
echo $(date -u)
echo "start download hyv model"
echo $(date -u)
mkdir -p workspace_data/models/diffusion_models && cd workspace_data/models/diffusion_models && git clone https://huggingface.co/Kijai/HunyuanVideo_comfy && cd ./../../..
mkdir -p workspace_data/models/vae
mv workspace_data/models/diffusion/HunyuanVideo_comfy/*vae* workspace_data/models/vae
echo "end download hyv model"
echo $(date -u)

echo "start download llava model for image prompt"
echo $(date -u)
mkdir -p workspace_data/models/LLM && cd workspace_data/models/LLM &&  git clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers && cd ./../../..
echo "end download llava model for image prompt"
echo $(date -u)

echo "start download clip model"
echo $(date -u)
mkdir -p workspace_data/models/clip && cd workspace_data/models/clip && git clone https://huggingface.co/openai/clip-vit-large-patch14 && cd ./../../..
echo "end download clip model"
echo $(date -u)

echo "start download llava model for text to video"
echo $(date -u)
mkdir -p workspace_data/models/LLM && cd workspace_data/models/LLM && git clone https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer && cd ./../../..
echo "end download llava model for text to video"
echo $(date -u)
