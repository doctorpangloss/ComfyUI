import base64
import io
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import torch
from PIL import Image

from comfy.component_model.tensor_types import RGBImageBatch
from comfy.language.language_types import LanguageModel, ProcessorResult, GENERATION_KWARGS_TYPE, TOKENS_TYPE, \
    TransformerStreamedProgress, LanguagePrompt
from comfy.utils import comfy_progress, ProgressBar, seed_for_block


def image_to_base64(image: RGBImageBatch) -> str:
    # Convert tensor to PIL Image
    pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Convert to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OpenAIStyleAPILanguageModelWrapper(LanguageModel, ABC):
    def __init__(self, model: str):
        self.model = model
        self.client = self._get_client()

    @abstractmethod
    def _get_client(self) -> Any:
        """Return the API client instance"""
        pass

    @abstractmethod
    def _prepare_messages(self, prompt: str, images: list[RGBImageBatch]) -> list[dict]:
        """Convert prompt and images into API-specific message format"""
        pass

    @abstractmethod
    def _handle_stream_chunk(self, chunk: Any) -> Optional[str]:
        """Extract text from a stream chunk in API-specific format"""
        pass

    def _validate_image_count(self, image_count: int) -> bool:
        """Validate the number of images being sent"""
        return True

    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None) -> "OpenAIStyleAPILanguageModelWrapper":
        return OpenAIStyleAPILanguageModelWrapper(ckpt_name)

    def generate(self, tokens: TOKENS_TYPE = None,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 0.0,
                 seed: int = 0,
                 sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                 *args,
                 **kwargs) -> str:
        sampler = sampler or {}
        prompt = tokens.get("inputs", [])
        prompt = "".join(prompt)
        images = tokens.get("images", [])

        # Validate image count
        if not self._validate_image_count(len(images)):
            raise ValueError(f"Invalid number of images for {self.__class__.__name__}")

        messages = self._prepare_messages(prompt, images)

        progress_bar: ProgressBar
        with comfy_progress(total=max_new_tokens) as progress_bar:
            token_count = 0
            full_response = ""

            def on_finalized_text(next_token: str, stop: bool):
                nonlocal token_count
                nonlocal progress_bar
                nonlocal full_response

                token_count += 1
                full_response += next_token
                preview = TransformerStreamedProgress(next_token=next_token)
                progress_bar.update_absolute(max_new_tokens if stop else token_count, total=max_new_tokens, preview_image_or_output=preview)

            with seed_for_block(seed):
                stream = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=sampler.get("temperature", 1.0),
                    top_p=sampler.get("top_p", 1.0),
                    stream=True
                )

                for chunk in stream:
                    if next_text := self._handle_stream_chunk(chunk):
                        on_finalized_text(next_text, False)

                on_finalized_text("", True)  # Signal the end of streaming

        return full_response

    def tokenize(self, prompt: str | LanguagePrompt, images: RGBImageBatch | None, chat_template: str | None = None) -> ProcessorResult:
        res: ProcessorResult = {
            "inputs": [prompt],
            "attention_mask": torch.ones(1, len(prompt)),  # Dummy attention mask
        }
        if images is not None:
            # this pattern keeps type checking
            res = {
                **res,
                "images": images,
                "pixel_values": images,
            }
        return res

    @property
    def repo_id(self) -> str:
        return f"{self.__class__.__name__.lower().replace('languagemodelwrapper', '')}/{self.model}"
