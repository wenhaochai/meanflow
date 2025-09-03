# Looped DiT: Universal Transformer version of DiT
# Reuses most components from models_dit.py

from functools import partial
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

# Import everything we need from existing modules
from models.models_dit import (
    DiTBlock, TimestepEmbedder, LabelEmbedder, FinalLayer, 
    get_2d_sincos_pos_embed, DiT  # Import DiT for unpatchify method
)
from models.timm_models import PatchEmbed

class LoopedDiT(nn.Module):
    """
    Segmented Looped version of DiT with loop-step conditioning.

    Design:
    - Prefix (p): independent blocks executed once at the beginning.
    - Shared (s) × Loops (l): a set of blocks executed l times with parameter sharing.
    - Suffix (e): independent blocks executed once at the end.

    t_loop (loop-step embedding):
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
    n_prefix: int = 3        # Number of prefix blocks (independent)
    n_shared: int = 3        # Number of shared blocks for looping
    n_loops: int = 2         # Number of times to loop the shared blocks
    n_suffix: int = 3        # Number of suffix blocks (independent)
    num_heads: int = 12
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    
    def setup(self):
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        # Reuse all embedders from original DiT
        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.h_embedder = TimestepEmbedder(self.hidden_size)
        # Loop-step embedder for per-iteration conditioning (t_loop)
        # Note: we re-use the same embedding machinery (sinusoidal + small MLP)
        # to keep scales compatible with other condition signals.
        self.l_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(get_2d_sincos_pos_embed(self.hidden_size, int(num_patches ** 0.5))).astype(jnp.float32)

        # Prefix blocks (independent)
        if self.n_prefix > 0:
            self.prefix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) 
                for _ in range(self.n_prefix)
            ])
        
        # Shared blocks for looping
        if self.n_shared > 0:
            self.shared_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.n_shared)
            ])
        
        # Suffix blocks (independent) 
        if self.n_suffix > 0:
            self.suffix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.n_suffix)
            ])
        
        # Reuse final layer
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x):
        """Reuse unpatchify logic from DiT"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def __call__(self, x, t, h, y, train=False, key=None):
        """
        Segmented looped forward pass - same interface as DiT.

        Inputs
        - x: (B, H, W, C) input
        - t: (B,) diffusion timestep (or any continuous-time signal)
        - h: (B,) auxiliary time-like signal (e.g., step size/interval)
        - y: (B,) class labels
        - train/key: as in the base DiT

        Conditioning
        - Base condition: c = t_emb + h_emb + y_emb
        - Stage/iteration condition: c_stage = c + l_emb(t_loop)
          where t_loop encodes the current stage: prefix=1.0, loops∈[1,0], suffix=0.0
        """
        # Same initial processing as original DiT
        x = self.x_embedder(x) + self.pos_embed_func()
        t = self.t_embedder(t)
        h = self.h_embedder(h)
        y = self.y_embedder(y, train=train, rng=key)
        c = t + h + y  # base condition

        # Small helper to embed a scalar t_loop ∈ [0, 1] for the whole batch
        bsz = x.shape[0]
        def loop_emb_from_scalar(scalar):
            s_vec = jnp.full((bsz,), float(scalar), dtype=jnp.float32)
            return self.l_embedder(s_vec)

        # 1. Apply prefix blocks (independent)
        if self.n_prefix > 0:
            # Use t_loop = 1.0 to denote the "early"/prefix phase (denoise-aligned)
            c_prefix = c + loop_emb_from_scalar(1.0)
            for block in self.prefix_blocks.layers:
                x = block(x, c_prefix)
        
        # 2. Apply shared blocks in loops 
        if self.n_shared > 0 and self.n_loops > 0:
            # Linearly sweep t_loop ∈ [1, 0] across loop iterations.
            # When n_loops==1, we set t_loop=1.0.
            denom = max(self.n_loops - 1, 1)
            for loop_idx in range(self.n_loops):
                t_loop_scalar = 1.0 - (loop_idx / denom)
                c_loop = c + loop_emb_from_scalar(t_loop_scalar)
                for block in self.shared_blocks.layers:
                    x = block(x, c_loop)
        
        # 3. Apply suffix blocks (independent)
        if self.n_suffix > 0:
            # Use t_loop = 0.0 to denote the "late"/suffix phase (denoise-aligned)
            c_suffix = c + loop_emb_from_scalar(0.0)
            for block in self.suffix_blocks.layers:
                x = block(x, c_suffix)
        
        # Same final processing
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

# Looped DiT configs - num_loops can be overridden via config
# Default num_loops values, but configurable
LoopedDiT_XL_2 = partial(LoopedDiT, hidden_size=1152, patch_size=2, num_heads=16)
LoopedDiT_XL_4 = partial(LoopedDiT, hidden_size=1152, patch_size=4, num_heads=16)
LoopedDiT_XL_8 = partial(LoopedDiT, hidden_size=1152, patch_size=8, num_heads=16)
LoopedDiT_L_2 = partial(LoopedDiT, hidden_size=1024, patch_size=2, num_heads=16)
LoopedDiT_L_4 = partial(LoopedDiT, hidden_size=1024, patch_size=4, num_heads=16)
LoopedDiT_L_8 = partial(LoopedDiT, hidden_size=1024, patch_size=8, num_heads=16)
LoopedDiT_B_2 = partial(LoopedDiT, hidden_size=768, patch_size=2, num_heads=12)
LoopedDiT_B_4 = partial(LoopedDiT, hidden_size=768, patch_size=4, num_heads=12)
LoopedDiT_B_8 = partial(LoopedDiT, hidden_size=768, patch_size=8, num_heads=12)
LoopedDiT_S_2 = partial(LoopedDiT, hidden_size=384, patch_size=2, num_heads=6)
LoopedDiT_S_4 = partial(LoopedDiT, hidden_size=384, patch_size=4, num_heads=6)
LoopedDiT_S_8 = partial(LoopedDiT, hidden_size=384, patch_size=8, num_heads=6)

LoopedDiT_models = {
    'LoopedDiT-XL/2': LoopedDiT_XL_2,  'LoopedDiT-XL/4': LoopedDiT_XL_4,  'LoopedDiT-XL/8': LoopedDiT_XL_8,
    'LoopedDiT-L/2':  LoopedDiT_L_2,   'LoopedDiT-L/4':  LoopedDiT_L_4,   'LoopedDiT-L/8':  LoopedDiT_L_8,
    'LoopedDiT-B/2':  LoopedDiT_B_2,   'LoopedDiT-B/4':  LoopedDiT_B_4,   'LoopedDiT-B/8':  LoopedDiT_B_8,
    'LoopedDiT-S/2':  LoopedDiT_S_2,   'LoopedDiT-S/4':  LoopedDiT_S_4,   'LoopedDiT-S/8':  LoopedDiT_S_8,
}

if __name__ == '__main__':
    import jax
    # Test the looped model
    model = LoopedDiT_S_4(num_classes=1000)
    variables = model.init(random.PRNGKey(0), 
                          jnp.ones((2, 32, 32, 4)), 
                          jnp.ones((2,)), 
                          jnp.ones((2,)), 
                          jnp.ones((2,), dtype=jnp.int32))
    
    x = np.random.randn(2, 32, 32, 4).astype(np.float32)
    t = np.random.randint(0, 1000, (2,)).astype(np.float32)
    h = np.random.randint(0, 1000, (2,)).astype(np.float32)
    y = np.random.randint(0, 1000, (2,)).astype(np.int32)

    out = model.apply(variables, x, t, h, y=y, train=True, key=random.PRNGKey(0))
    print(f"Output range: [{out.min():.6f}, {out.max():.6f}]")
    print(f"Output shape: {out.shape}")
    
    # Compare parameter count
    param_count = sum(x.size for x in jax.tree.leaves(variables['params']))
    print(f"LoopedDiT-S/4 parameter count: {param_count:,}")
