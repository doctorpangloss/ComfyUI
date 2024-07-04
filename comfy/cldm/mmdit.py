import torch
from typing import Dict, Optional, List

from einops import einops
from torch import Tensor

from ..ldm.modules.diffusionmodules.mmdit import MMDiT, PatchEmbed


def default(x, y):
    if x is not None:
        return x
    return y

class ControlNet(MMDiT):
    def __init__(
        self,
        num_blocks = None,
        dtype = None,
        device = None,
        operations = None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, device=device, operations=operations, final_layer=False, num_blocks=num_blocks, **kwargs)
        # controlnet_blocks
        self.controlnet_blocks = torch.nn.ModuleList([])
        for _ in range(len(self.joint_blocks)):
            self.controlnet_blocks.append(operations.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype))

        self.pos_embed_input = PatchEmbed(
            None,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=False,
            dtype=dtype,
            device=device,
            operations=operations
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hint = None,
    ) -> Dict[str, List[Tensor]]:

        #weird sd3 controlnet specific stuff
        y = torch.zeros_like(y)

        if self.context_processor is not None:
            context = self.context_processor(context)

        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw, device=x.device).to(dtype=x.dtype, device=x.device)
        x += self.pos_embed_input(hint)

        c = self.t_embedder(timesteps, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)
            c = c + y

        if context is not None:
            context = self.context_embedder(context)

        if self.register_length > 0:
            context = torch.cat(
                (
                    einops.repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    default(context, torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        output = []

        blocks = len(self.joint_blocks)
        for i in range(blocks):
            context, x = self.joint_blocks[i](
                context,
                x,
                c=c,
                use_checkpoint=self.use_checkpoint,
            )

            out = self.controlnet_blocks[i](x)
            count = self.depth // blocks
            if i == blocks - 1:
                count -= 1
            for j in range(count):
                output.append(out)

        return {"output": output}