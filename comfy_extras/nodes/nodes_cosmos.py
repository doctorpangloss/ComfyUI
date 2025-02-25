import torch

import comfy.model_management
import comfy.utils
from comfy.language.language_types import LanguageModel
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.common import MAX_RESOLUTION
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy_extras.nodes.nodes_language import TransformersLoader, OneShotInstructTokenize, _AUTO_CHAT_TEMPLATE


class EmptyCosmosLatentVideo(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"width": ("INT", {"default": 1280, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 704, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 121, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples": latent},)


def vae_encode_with_padding(vae, image, width, height, length, padding=0):
    pixels = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    pixel_len = min(pixels.shape[0], length)
    padded_length = min(length, (((pixel_len - 1) // 8) + 1 + padding) * 8 - 7)
    padded_pixels = torch.ones((padded_length, height, width, 3)) * 0.5
    padded_pixels[:pixel_len] = pixels[:pixel_len]
    latent_len = ((pixel_len - 1) // 8) + 1
    latent_temp = vae.encode(padded_pixels)
    return latent_temp[:, :, :latent_len]


class CosmosImageToVideoLatent(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",),
                             "width": ("INT", {"default": 1280, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 704, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 121, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             },
                "optional": {"start_image": ("IMAGE",),
                             "end_image": ("IMAGE",),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, vae, width, height, length, batch_size, start_image=None, end_image=None):
        latent = torch.zeros([1, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is None and end_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return (out_latent,)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 8) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            latent_temp = vae_encode_with_padding(vae, start_image, width, height, length, padding=1)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        if end_image is not None:
            latent_temp = vae_encode_with_padding(vae, end_image, width, height, length, padding=0)
            latent[:, :, -latent_temp.shape[-3]:] = latent_temp
            mask[:, :, -latent_temp.shape[-3]:] *= 0.0

        out_latent = {}
        out_latent["samples"] = latent.repeat((batch_size,) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size,) + (1,) * (mask.ndim - 1))
        return (out_latent,)


class CosmosPromptUpsamplerTransformersLoader(TransformersLoader):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": ("STRING", {}),
            },
        }


# from https://github.com/NVIDIA/Cosmos/blob/b867572b99d08f450ddb8bcd6661d8c35bf6b967/cosmos1/models/diffusion/nemo/inference/inference_utils.py#L54
FROM_COSMOS_REPO_PROMPT_PREFIX = "Upsample the short caption to a long caption: "


class CosmosUpsamplePromptTokenize(OneShotInstructTokenize):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def execute(self, model: LanguageModel, prompt: str, images: list[torch.Tensor] | torch.Tensor = None, chat_template: str = "__auto__") -> ValidatedNodeResult:
        return super().execute(model, f"{FROM_COSMOS_REPO_PROMPT_PREFIX}{prompt}", images=None, chat_template=_AUTO_CHAT_TEMPLATE)


export_custom_nodes()
