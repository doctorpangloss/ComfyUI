from __future__ import annotations

import dataclasses
import logging
import math
import os.path
from enum import Enum
from typing import Any, Optional

import torch
import yaml

from . import clip_vision
from . import diffusers_convert
from . import gligen
from . import lora
from . import model_detection
from . import model_management
from . import model_patcher
from . import model_sampling
from . import sd1_clip
from . import sdxl_clip
from . import utils
from .hooks import EnumHookMode
from .ldm.audio.autoencoder import AudioOobleckVAE
from .ldm.cascade.stage_a import StageA
from .ldm.cascade.stage_c_coder import StageC_coder
from .ldm.cosmos.vae import CausalContinuousVideoTokenizer
from .ldm.flux.redux import ReduxImageEncoder
from .ldm.genmo.vae import model as genmo_model
from .ldm.lightricks.vae import causal_video_autoencoder as lightricks
from .ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
from .lora_convert import convert_lora
from .model_management import load_models_gpu
from .t2i_adapter import adapter
from .taesd import taesd
from .text_encoders import aura_t5
from .text_encoders import cosmos
from .text_encoders import flux
from .text_encoders import genmo
from .text_encoders import hunyuan_video
from .text_encoders import hydit
from .text_encoders import long_clipl
from .text_encoders import lt
from .text_encoders import lumina2
from .text_encoders import pixart_t5
from .text_encoders import sa_t5
from .text_encoders import sd2_clip
from .text_encoders import sd3_clip
from .utils import ProgressBar

logger = logging.getLogger(__name__)


def load_lora_for_models(model, clip, _lora, strength_model, strength_clip):
    key_map = {}
    if model is not None:
        key_map = lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    _lora = convert_lora(_lora)
    loaded = lora.load_lora(_lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logger.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, target: CLIPTarget = None, embedding_directory=None, no_init=False, textmodel_json_config=None, tokenizer_data: dict | None = None, parameters=0, model_options={}):
        if tokenizer_data is None:
            tokenizer_data = dict()
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", model_management.text_encoder_device())
        offload_device = model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management.text_encoder_dtype(load_device)

        params['dtype'] = dtype
        params['device'] = model_options.get("initial_device", model_management.text_encoder_initial_device(load_device, offload_device, parameters * model_management.dtype_size(dtype)))
        if "textmodel_json_config" not in params and textmodel_json_config is not None:
            params['textmodel_json_config'] = textmodel_json_config

        params['model_options'] = model_options

        self.cond_stage_model = clip(**(params))

        for dt in self.cond_stage_model.dtypes:
            if not model_management.supports_cast(load_device, dt):
                load_device = offload_device
                if params['device'] != offload_device:
                    self.cond_stage_model.to(offload_device)
                    logger.warning("Had to shift TE back.")

        self.tokenizer: "sd1_clip.SD1Tokenizer" = tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.patcher = model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.patcher.hook_mode = EnumHookMode.MinVram
        self.patcher.is_clip = True
        self.apply_hooks_to_conds = None
        if params['device'] == load_device:
            model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.layer_idx = None
        self.use_clip_schedule = False
        logger.debug("CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(load_device, offload_device, params['device'], dtype))

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        # cloning the tokenizer allows the vocab updates to work more idiomatically
        n.tokenizer = self.tokenizer.clone()
        n.layer_idx = self.layer_idx
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def add_hooks_to_dict(self, pooled_dict: dict[str]):
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds
        return pooled_dict

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict[str] = {}, show_pbar=True):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
            cond = pooled_dict.pop("cond")
            # add/update any keys with the provided add_dict
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model()
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            for scheduled_opts in scheduled_keyframes:
                t_range = scheduled_opts[0]
                # don't bother encoding any conds outside of start_percent and end_percent bounds
                if "start_percent" in add_dict:
                    if t_range[1] < add_dict["start_percent"]:
                        continue
                if "end_percent" in add_dict:
                    if t_range[0] > add_dict["end_percent"]:
                        continue
                hooks_keyframes = scheduled_opts[1]
                for hook, keyframe in hooks_keyframes:
                    hook.hook_keyframe._current_keyframe = keyframe
                # apply appropriate hooks with values that match new hook_keyframe
                self.patcher.patch_hooks(all_hooks)
                # perform encoding as normal
                o = self.cond_stage_model.encode_token_weights(tokens)
                cond, pooled = o[:2]
                pooled_dict = {"pooled_output": pooled}
                # add clip_start_percent and clip_end_percent in pooled
                pooled_dict["clip_start_percent"] = t_range[0]
                pooled_dict["clip_end_percent"] = t_range[1]
                # add/update any keys with the provided add_dict
                pooled_dict.update(add_dict)
                # add hooks stored on clip
                self.add_hooks_to_dict(pooled_dict)
                all_cond_pooled.append([cond, pooled_dict])
                if show_pbar:
                    pbar.update(1)
                model_management.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self):
        load_models_gpu([self.patcher])
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None):
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys():  # diffusers format
            sd = diffusers_convert.convert_vae_state_dict(sd)

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype)  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]

        self.downscale_index_formula = None
        self.upscale_index_formula = None

        if config is None:
            if "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                            encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': encoder_config},
                                                            decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder", 'params': decoder_config})
            elif "taesd_decoder.1.weight" in sd:
                self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
                self.first_stage_model = taesd.TAESD(latent_channels=self.latent_channels)
            elif "vquantizer.codebook.weight" in sd:  # VQGan: stage a of stable cascade
                self.first_stage_model = StageA()
                self.downscale_ratio = 4
                self.upscale_ratio = 4
                # TODO
                # self.memory_used_encode
                # self.memory_used_decode
                self.process_input = lambda image: image
                self.process_output = lambda image: image
            elif "backbone.1.0.block.0.1.num_batches_tracked" in sd:  # effnet: encoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["encoder.{}".format(k)] = sd[k]
                sd = new_sd
            elif "blocks.11.num_batches_tracked" in sd:  # previewer: decoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["previewer.{}".format(k)] = sd[k]
                sd = new_sd
            elif "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd:  # combined effnet and previewer for stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
            elif "decoder.conv_in.weight" in sd:
                # default SD1.x/SD2.x VAE parameters
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

                if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd:  # Stable diffusion x4 upscaler VAE
                    ddconfig['ch_mult'] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                if 'post_quant_conv.weight' in sd:
                    self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                else:
                    self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
                                                                decoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Decoder", 'params': ddconfig})
            elif "decoder.layers.1.layers.0.beta" in sd:
                self.first_stage_model = AudioOobleckVAE()
                self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * 2048) * model_management.dtype_size(dtype)
                self.latent_channels = 64
                self.output_channels = 2
                self.upscale_ratio = 2048
                self.downscale_ratio = 2048
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            elif "blocks.2.blocks.3.stack.5.weight" in sd or "decoder.blocks.2.blocks.3.stack.5.weight" in sd or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd:  # genmo mochi vae
                if "blocks.2.blocks.3.stack.5.weight" in sd:
                    sd = utils.state_dict_prefix_replace(sd, {"": "decoder."})
                if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                    sd = utils.state_dict_prefix_replace(sd, {"": "encoder."})
                self.first_stage_model = genmo_model.VideoVAE()
                self.latent_channels = 12
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
                self.upscale_index_formula = (6, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
                self.downscale_index_formula = (6, 8, 8)
                self.working_dtypes = [torch.float16, torch.float32]
            elif "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd:  # lightricks ltxv
                tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
                version = 0
                if tensor_conv1.shape[0] == 512:
                    version = 0
                elif tensor_conv1.shape[0] == 1024:
                    version = 1
                self.first_stage_model = lightricks.VideoVAE(version=version)
                self.latent_channels = 128
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (900 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (70 * max(shape[2], 7) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                self.upscale_index_formula = (8, 32, 32)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
                self.downscale_index_formula = (8, 32, 32)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.conv_in.conv.weight" in sd:
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                ddconfig["conv3d"] = True
                ddconfig["time_compress"] = 4
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                self.memory_used_decode = lambda shape, dtype: (1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (900 * max(shape[2], 2) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            elif "decoder.unpatcher3d.wavelets" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
                self.upscale_index_formula = (8, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
                self.downscale_index_formula = (8, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {'z_channels': 16, 'latent_channels': self.latent_channels, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3, 'channels': 128, 'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0, 'patch_size': 4, 'num_groups': 1, 'temporal_compression': 8, 'spacial_compression': 8}
                self.first_stage_model = CausalContinuousVideoTokenizer(**ddconfig)
                # TODO: these values are a bit off because this is not a standard VAE
                self.memory_used_decode = lambda shape, dtype: (50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            else:
                logger.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logger.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logger.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        logger.debug("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))

    def vae_encode_crop_pixels(self, pixels):
        downscale_ratio = self.spacial_compression_encode()

        dims = pixels.shape[1:-1]
        for d in range(len(dims)):
            x = (dims[d] // downscale_ratio) * downscale_ratio
            x_offset = (dims[d] % downscale_ratio) // 2
            if x != dims[d]:
                pixels = pixels.narrow(d + 1, x_offset, x)
        return pixels

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = utils.ProgressBar(steps)

        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        output = self.process_output(
            (utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar) +
             utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar) +
             utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar))
            / 3.0)
        return output

    def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        return self.process_output(utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device))

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        return self.process_output(utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = utils.ProgressBar(steps)

        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        samples = utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples += utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples += utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples /= 3.0
        return samples

    def encode_tiled_1d(self, samples, tile_x=128 * 2048, overlap=32 * 2048):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        return utils.tiled_scale_multidim(samples, encode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)

    def encode_tiled_3d(self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        return utils.tiled_scale_multidim(samples, encode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.downscale_ratio, out_channels=self.latent_channels, downscale=True, index_formulas=self.downscale_index_formula, output_device=self.output_device)

    def decode(self, samples_in):
        pixel_samples = None
        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
                out = self.process_output(self.first_stage_model.decode(samples).to(self.output_device).float())
                if pixel_samples is None:
                    pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
                pixel_samples[x:x + batch_number] = out
        except model_management.OOM_EXCEPTION:
            logger.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            dims = samples_in.ndim - 2
            if dims == 1:
                pixel_samples = self.decode_tiled_1d(samples_in)
            elif dims == 2:
                pixel_samples = self.decode_tiled_(samples_in)
            elif dims == 3:
                tile = 256 // self.spacial_compression_decode()
                overlap = tile // 4
                pixel_samples = self.decode_tiled_3d(samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        memory_used = self.memory_used_decode(samples.shape, self.vae_dtype)  # TODO: calculate mem required for tile
        load_models_gpu([self.patcher], memory_required=memory_used)
        dims = samples.ndim - 2
        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        if dims == 1:
            args.pop("tile_y")
            output = self.decode_tiled_1d(samples, **args)
        elif dims == 2:
            output = self.decode_tiled_(samples, **args)
        elif dims == 3:
            if overlap_t is None:
                args["overlap"] = (1, overlap, overlap)
            else:
                args["overlap"] = (max(1, overlap_t), overlap, overlap)
            if tile_t is not None:
                args["tile_t"] = max(2, tile_t)

            output = self.decode_tiled_3d(samples, **args)
        else:
            raise ValueError(f"invalid dims={dims}")
        return output.movedim(1, -1)

    def encode(self, pixel_samples):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / max(1, memory_used))
            batch_number = max(1, batch_number)
            samples = None
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = self.process_input(pixel_samples[x:x + batch_number]).to(self.vae_dtype).to(self.device)
                out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
                if samples is None:
                    samples = torch.empty((pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
                samples[x:x + batch_number] = out

        except model_management.OOM_EXCEPTION:
            logger.warning("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            if self.latent_dim == 3:
                tile = 256
                overlap = tile // 4
                samples = self.encode_tiled_3d(pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
            elif self.latent_dim == 1:
                samples = self.encode_tiled_1d(pixel_samples)
            else:
                samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        dims = self.latent_dim
        pixel_samples = pixel_samples.movedim(-1, 1)
        if dims == 3:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)  # TODO: calculate mem required for tile
        load_models_gpu([self.patcher], memory_required=memory_used)

        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        samples = None
        if dims == 1:
            args.pop("tile_y")
            samples = self.encode_tiled_1d(pixel_samples, **args)
        elif dims == 2:
            samples = self.encode_tiled_(pixel_samples, **args)
        elif dims == 3:
            if tile_t is not None:
                tile_t_latent = max(2, self.downscale_ratio[0](tile_t))
            else:
                tile_t_latent = 9999
            args["tile_t"] = self.upscale_ratio[0](tile_t_latent)

            if overlap_t is None:
                args["overlap"] = (1, overlap, overlap)
            else:
                args["overlap"] = (self.upscale_ratio[0](max(1, min(tile_t_latent // 2, self.downscale_ratio[0](overlap_t)))), overlap, overlap)
            maximum = pixel_samples.shape[2]
            maximum = self.upscale_ratio[0](self.downscale_ratio[0](maximum))

            samples = self.encode_tiled_3d(pixel_samples[:, :, :maximum], **args)
        else:
            raise ValueError(f"unsupported values dim {dims}")

        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()

    def spacial_compression_decode(self):
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio

    def spacial_compression_encode(self):
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio

    def temporal_compression_decode(self):
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None


class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)


def load_style_model(ckpt_path):
    model_data = utils.load_torch_file(ckpt_path, safe_load=True)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = adapter.StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8)
    elif "redux_down.weight" in keys:
        model = ReduxImageEncoder()
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12


@dataclasses.dataclass
class CLIPTarget:
    clip: Optional[Any] = None
    vae: Optional[Any] = None
    params: Optional[dict] = dataclasses.field(default_factory=dict)
    tokenizer: Optional[Any] = None


def load_clip(ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, textmodel_json_config: str | dict | None = None, model_options=None):
    if model_options is None:
        model_options = dict()
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options, textmodel_json_config=textmodel_json_config)


class TEModel(Enum):
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9


def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
        elif weight.shape[-1] == 2048:
            return TEModel.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        return TEModel.T5_BASE
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TEModel.GEMMA_2_2B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TEModel.LLAMA3_8
    return None


def t5xxl_detect(clip_data):
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"

    for sd in clip_data:
        if weight_name in sd or weight_name_old in sd:
            return sd3_clip.t5_xxl_detect(sd)

    return {}


def llama_detect(clip_data):
    weight_name = "model.layers.0.self_attn.k_proj.weight"

    for sd in clip_data:
        if weight_name in sd:
            return hunyuan_video.llama_detect(sd)

    return {}


def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}, textmodel_json_config=None):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1)  # old models saved with the CLIPSave node

    tokenizer_data = {}
    clip_target = CLIPTarget()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=False, clip_g=True, t5=False)
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = sd2_clip.SD2ClipModel
            clip_target.tokenizer = sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data))
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = lt.ltxv_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART:
                clip_target.clip = pixart_t5.pixart_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = pixart_t5.PixArtTokenizer
            else:  # CLIPType.MOCHI
                clip_target.clip = genmo.mochi_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = cosmos.te(**t5xxl_detect(clip_data))
            clip_target.tokenizer = cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = aura_t5.AuraT5Model
            clip_target.tokenizer = aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            clip_target.clip = sa_t5.SAT5Model
            clip_target.tokenizer = sa_t5.SAT5Tokenizer
        elif te_model == TEModel.GEMMA_2_2B:
            clip_target.clip = lumina2.te(**llama_detect(clip_data))
            clip_target.tokenizer = lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        else:
            if clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=True, clip_g=False, t5=False)
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = sd3_clip.sd3_clip(clip_l=TEModel.CLIP_L in te_models, clip_g=TEModel.CLIP_G in te_models, t5=TEModel.T5_XXL in te_models, **t5xxl_detect(clip_data))
            clip_target.tokenizer = sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = hydit.HyditModel
            clip_target.tokenizer = hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            clip_target.clip = flux.flux_clip(**t5xxl_detect(clip_data))
            clip_target.tokenizer = flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = hunyuan_video.hunyuan_video_clip(**llama_detect(clip_data))
            clip_target.tokenizer = hunyuan_video.HunyuanVideoTokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = sd3_clip.sd3_clip(**t5xxl_detect(clip_data))
        clip_target.tokenizer = sd3_clip.SD3Tokenizer

    parameters = 0
    for c in clip_data:
        parameters += utils.calculate_parameters(c)
        tokenizer_data, model_options = long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = CLIP(clip_target, embedding_directory=embedding_directory, textmodel_json_config=textmodel_json_config, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logger.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logger.debug("clip unexpected: {}".format(u))
    return clip


def load_gligen(ckpt_path):
    data = utils.load_torch_file(ckpt_path, safe_load=True)
    model = gligen.load_gligen(data)
    if model_management.should_use_fp16():
        model = model.half()
    return model_patcher.ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device())


def load_checkpoint(config_path=None, ckpt_path=None, output_vae=True, output_clip=True, embedding_directory=None, state_dict=None, config=None):
    logger.warning("Warning: The load checkpoint with config function is deprecated and will eventually be removed, please use the other one.")
    model, clip, vae, _ = load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, output_clipvision=False, embedding_directory=embedding_directory, output_model=True)
    # TODO: this function is a mess and should be removed eventually
    if config is None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            m = model.clone()

            class ModelSamplingAdvanced(model_sampling.ModelSamplingDiscrete, model_sampling.V_PREDICTION):
                pass

            m.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
            model = m

    layer_idx = clip_config.get("params", {}).get("layer_idx", None)
    if layer_idx is not None:
        clip.clip_layer(layer_idx)

    return (model, clip, vae)


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options=None, te_model_options=None):
    if te_model_options is None:
        te_model_options = {}
    if model_options is None:
        model_options = {}
    sd = utils.load_torch_file(ckpt_path)
    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options, ckpt_path=ckpt_path)
    if out is None:
        raise RuntimeError("Could not detect model type of: {}".format(ckpt_path))
    return out


def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options=None, te_model_options=None, ckpt_path=""):
    if te_model_options is None:
        te_model_options = {}
    if model_options is None:
        model_options = {}
    clip = None
    clipvision = None
    vae = None
    model = None
    _model_patcher = None
    inital_load_device = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
    if model_config is None:
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_initial_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logger.warning("clip missing: {}".format(m))
                    else:
                        logger.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logger.debug("clip unexpected {}:".format(u))
            else:
                logger.warning(f"no CLIP/text encoder weights in checkpoint {ckpt_path}, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logger.debug("left over keys: {}".format(left_over))

    if output_model:
        _model_patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device(), ckpt_name=os.path.basename(ckpt_path))
        if inital_load_device != torch.device("cpu"):
            model_management.load_models_gpu([_model_patcher], force_full_load=True)

    return (_model_patcher, clip, vae, clipvision)


def load_diffusion_model_state_dict(sd, model_options: dict = None, ckpt_path: Optional[str] = ""):  # load unet in diffusers or regular format
    if model_options is None:
        model_options = {}
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = utils.calculate_parameters(sd)
    weight_dtype = utils.weight_dtype(sd)
    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logger.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logger.info("left over keys in unet: {}".format(left_over))
    return model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device, ckpt_name=os.path.basename(ckpt_path))


def load_diffusion_model(unet_path, model_options: dict = None):
    if model_options is None:
        model_options = {}
    sd = utils.load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(sd, model_options=model_options, ckpt_path=unet_path)
    if model is None:
        logger.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
    return model


def load_unet(unet_path, dtype=None):
    logging.warning("The load_unet function has been deprecated and will be removed please switch to: load_diffusion_model")
    return load_diffusion_model(unet_path, model_options={"dtype": dtype})


def load_unet_state_dict(sd, dtype=None):
    logging.warning("The load_unet_state_dict function has been deprecated and will be removed please switch to: load_diffusion_model_state_dict")
    return load_diffusion_model_state_dict(sd, model_options={"dtype": dtype})


def save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    model_management.load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    utils.save_torch_file(sd, output_path, metadata=metadata)
