# Reuses most components from models_dit.py

from functools import partial
from typing import Optional
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

from models.models_dit import (
    DiTBlock, TimestepEmbedder, LabelEmbedder, FinalLayer, 
    get_2d_sincos_pos_embed, DiT
)
from models.torch_models import TorchLinear
from models.timm_models import PatchEmbed

class LoopedDiT(nn.Module):
    """
    Design:
    - Prefix (p): independent blocks executed once at the beginning.
    - Shared (s) for Loops (l): a set of blocks executed l times with parameter sharing.
    - Suffix (e): independent blocks executed once at the end.

    loop-step embedding:
    - We inject a scalar loop-step "time" per stage/iteration and embed it via the same
      TimestepEmbedder used elsewhere. The embedded vector is added into the global
      condition c, which feeds adaLN modulation (LN/FiLM) in each block.
    - Convention (aligned with denoise from 1→0): prefix uses t_loop=1.0, suffix uses t_loop=0.0.
      For the shared loops, t_loop linearly spans [1, 0] over loop_idx ∈ [0, l-1].
    - This enables the shared block to specialize across iterations without adding
      parameters, improving phase separation and few-step quality.
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 768
    n_prefix: int = 0          # Number of prefix blocks (independent)
    n_shared: int = 12         # Number of shared blocks for looping
    n_loops: int = 2           # Number of times to loop the shared blocks
    n_suffix: int = 0          # Number of suffix blocks (independent)
    num_heads: int = 12
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    loop_emb_max: float = 1.0  # Maximum loop embedding value (first loop)
    loop_emb_min: float = 0.0  # Minimum loop embedding value (last loop)
    # 'token': patchify + pos-embed once, loop in token space (the Sept-Oct 2025 p/s/e runs).
    # 'pixel': re-patchify / final_layer / unpatchify inside every loop (Nov 2025 p0s12 variant;
    #          requires n_prefix == n_suffix == 0).
    loop_mode: str = 'token'
    # MLP ratio of the shared (looped) blocks only; None -> mlp_ratio. Used by the dense
    # control for the MoE top-2 comparison (mlp_ratio 8 == two active experts of ratio 4).
    shared_mlp_ratio: Optional[float] = None
    
    def setup(self):
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.h_embedder = TimestepEmbedder(self.hidden_size)
        self.l_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(get_2d_sincos_pos_embed(self.hidden_size, int(num_patches ** 0.5))).astype(jnp.float32)

        if self.n_prefix > 0:
            self.prefix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) 
                for _ in range(self.n_prefix)
            ])
        
        if self.n_shared > 0:
            shared_mlp_ratio = self.mlp_ratio if self.shared_mlp_ratio is None else self.shared_mlp_ratio
            self.shared_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=shared_mlp_ratio)
                for _ in range(self.n_shared)
            ])
        
        if self.n_suffix > 0:
            self.suffix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.n_suffix)
            ])
        
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        ---
        return: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def get_loop_emb(self, loop_idx, batch_size):
        """Return base condition adjusted with loop-step embedding for this iteration."""
        # Linearly sweep t_loop ∈ [loop_emb_max, loop_emb_min] across loop iterations (only if loop_emb enabled).
        # When n_loops==1, we set t_loop=loop_emb_max.
        denom = max(self.n_loops - 1, 1) # keep original n_loops to control the loop-embedding range
        range_span = self.loop_emb_max - self.loop_emb_min
        t_loop_scalar = self.loop_emb_max - (loop_idx / denom) * range_span
        loop_scalar_vec = jnp.full((batch_size,), float(t_loop_scalar), dtype=jnp.float32)
        return self.l_embedder(loop_scalar_vec)

    def __call__(self, x, t, h, y, train=False, key=None):
        t = self.t_embedder(t)
        h = self.h_embedder(h)
        y = self.y_embedder(y, train=train, rng=key)
        c = t + h + y
        batch_size = x.shape[0]

        if self.loop_mode == 'token':
            # Same initial processing as DiT: patchify + positional embedding once.
            x = self.x_embedder(x) + self.pos_embed_func()

            if self.n_prefix > 0:
                for block in self.prefix_blocks.layers:
                    x = block(x, c)

            if self.n_shared > 0:
                for loop_idx in range(self.n_loops):
                    c_loop = c + self.get_loop_emb(loop_idx, batch_size)
                    for block in self.shared_blocks.layers:
                        x = block(x, c_loop)

            if self.n_suffix > 0:
                for block in self.suffix_blocks.layers:
                    x = block(x, c)

            x = self.final_layer(x, c)
            x = self.unpatchify(x)
            return x

        elif self.loop_mode == 'pixel':
            assert self.n_prefix == 0 and self.n_suffix == 0, \
                "loop_mode='pixel' loops in image space; prefix/suffix blocks are not supported"
            for loop_idx in range(self.n_loops):
                x = self.x_embedder(x)
                x = x + self.pos_embed_func()
                c_loop = c + self.get_loop_emb(loop_idx, batch_size)
                for block in self.shared_blocks.layers:
                    x = block(x, c_loop)
                x = self.final_layer(x, c)
                x = self.unpatchify(x)
            return x

        else:
            raise ValueError(f"Unknown loop_mode: {self.loop_mode}")

LoopedDiT_B_4 = partial(LoopedDiT, hidden_size=768, patch_size=4, num_heads=12, n_prefix=0, n_shared=12, n_loops=2, n_suffix=0)

LoopedDiT_models = {
    'LoopedDiT-B/4':  LoopedDiT_B_4
}
