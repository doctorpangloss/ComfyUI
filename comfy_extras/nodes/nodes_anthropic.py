import os
from typing import Optional, Any

from anthropic import Anthropic

from comfy.cli_args import args
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.language.language_types import LanguageModel
from comfy.language.openai_like_clients import OpenAIStyleAPILanguageModelWrapper, image_to_base64
from comfy.nodes.package_typing import CustomNode, InputTypes


class AnthropicClient:
    _client: Optional[Anthropic] = None

    @staticmethod
    def instance() -> Anthropic:
        if AnthropicClient._client is None:
            anthropic_api_key = args.anthropic_api_key
            AnthropicClient._client = Anthropic(
                api_key=anthropic_api_key,
            )
        return AnthropicClient._client


def validate_anthropic_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY", args.anthropic_api_key)
    if api_key is None or api_key == "":
        return "set ANTHROPIC_API_KEY environment variable"
    return True


class AnthropicLanguageModelWrapper(OpenAIStyleAPILanguageModelWrapper):
    def _get_client(self) -> Anthropic:
        return AnthropicClient.instance()

    def _validate_image_count(self, image_count: int) -> bool:
        # Anthropic Claude 3 supports up to 100 images for API requests
        return 0 <= image_count <= 100

    def _prepare_messages(self, prompt: str, images: list[RGBImageBatch]) -> list[dict]:
        content = []

        # According to docs, images work best when placed before text
        for image in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_to_base64(image)
                }
            })

        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })

        return [{
            "role": "user",
            "content": content
        }]

    def _handle_stream_chunk(self, chunk: Any) -> Optional[str]:
        return chunk.delta.text

    def repo_id(self) -> str:
        return f"anthropic/{self.model}"


class AnthropicLanguageModelLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": (
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20241022", "claude-3-opus-20241022"],
                    {"default": "claude-3-5-sonnet-20241022"}
                )
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("language model",)

    FUNCTION = "execute"
    CATEGORY = "anthropic"

    def execute(self, model: str) -> tuple[LanguageModel]:
        return AnthropicLanguageModelWrapper(model),

    @classmethod
    def VALIDATE_INPUTS(cls):
        return validate_anthropic_key()


NODE_CLASS_MAPPINGS = {
    "AnthropicLanguageModelLoader": AnthropicLanguageModelLoader
}
