# LoopedDiT-MoE: Mixture-of-Experts version of LoopedDiT
# Replaces MLP layers with MoE while keeping attention unchanged

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

# Reuse all base components from DiT
from models.models_dit import (
    TimestepEmbedder, LabelEmbedder, FinalLayer, DiTBlock,
    get_2d_sincos_pos_embed, modulate, unsqueeze,
    DiTLinear, DiTMlp  # Reuse DiT's linear and MLP
)
from models.timm_models import PatchEmbed
from models.torch_models import TorchLinear, TorchLayerNorm


#################################################################################
#                           MoE Components                                      #
#################################################################################

# Reuse DiTMlp as expert - no need to redefine


class MoEMlp(nn.Module):
    """
    Mixture of Experts MLP layer with optional shared experts.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension for each expert
        num_experts: Total number of routed experts
        num_active_experts: Number of experts to activate per token (top-k)
        num_shared_experts: Number of shared experts (always active)
        use_auxiliary_loss: Whether to compute load balancing loss
    """
    in_features: int
    hidden_features: int
    num_experts: int = 8
    num_active_experts: int = 1
    num_shared_experts: int = 0
    use_auxiliary_loss: bool = True

    def setup(self):
        # Create routed expert MLPs - reuse DiTMlp
        approx_gelu = lambda: partial(nn.gelu, approximate=True)
        self.experts = [
            DiTMlp(
                in_features=self.in_features,
                hidden_features=self.hidden_features,
                act_layer=approx_gelu,
                drop=0
            )
            for _ in range(self.num_experts)
        ]

        # Create shared experts (always active) if specified
        if self.num_shared_experts > 0:
            self.shared_experts = [
                DiTMlp(
                    in_features=self.in_features,
                    hidden_features=self.hidden_features,
                    act_layer=approx_gelu,
                    drop=0
                )
                for _ in range(self.num_shared_experts)
            ]

        # Router: maps input features to expert scores (only for routed experts)
        self.router = DiTLinear(self.in_features, self.num_experts, bias=False)

    def _top_k_routing(self, router_logits):
        """
        Select top-k experts for each token.

        Args:
            router_logits: (B, N, num_experts)

        Returns:
            top_k_indices: (B, N, k) - indices of selected experts
            top_k_probs: (B, N, k) - normalized probabilities
        """
        # Compute softmax over experts
        router_probs = jax.nn.softmax(router_logits, axis=-1)  # (B, N, num_experts)

        # Select top-k experts
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, k=self.num_active_experts)

        # Renormalize top-k probabilities
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)

        return top_k_indices, top_k_probs, router_probs

    def _compute_load_balance_loss(self, router_probs, top_k_indices):
        """
        Compute auxiliary load balancing loss to encourage uniform expert usage.

        Loss = num_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens assigned to expert i (hard assignment)
              P_i = average router probability for expert i (soft)

        Args:
            router_probs: (B, N, num_experts) - router probabilities
            top_k_indices: (B, N, k) - selected expert indices

        Returns:
            load_balance_loss: scalar
        """
        if not self.use_auxiliary_loss:
            return jnp.asarray(0.0, dtype=router_probs.dtype)

        # P_i: Average router probability per expert (soft assignment)
        mean_probs = jnp.mean(router_probs, axis=(0, 1))  # (num_experts,)

        # f_i: Fraction of tokens assigned to each expert (hard assignment)
        # Use one-hot encoding to count expert usage (more efficient than loop)
        B, N, k = top_k_indices.shape
        expert_mask = jax.nn.one_hot(top_k_indices, self.num_experts)  # (B, N, k, num_experts)
        expert_mask = jnp.sum(expert_mask, axis=2)  # (B, N, num_experts) - any token selecting expert
        expert_usage = jnp.mean(expert_mask, axis=(0, 1)) / self.num_active_experts  # (num_experts,)

        # Load balance loss: encourage uniform distribution
        loss = self.num_experts * jnp.sum(mean_probs * expert_usage)

        return loss

    def __call__(self, x):
        """
        Forward pass with expert routing and shared experts.

        Args:
            x: (B, N, C) input tensor

        Returns:
            output: (B, N, C) processed tensor
            aux_loss: scalar load balancing loss
        """
        B, N, C = x.shape

        # Initialize output
        output = jnp.zeros_like(x)

        # 1. Process through shared experts (always active)
        if self.num_shared_experts > 0:
            # Average output from all shared experts
            shared_output = jnp.zeros_like(x)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(x)
            # Average the shared expert outputs
            shared_output = shared_output / self.num_shared_experts
            output = output + shared_output

        # 2. Process through routed experts (top-k selection)
        # Compute router logits
        router_logits = self.router(x)  # (B, N, num_experts)

        # Top-k expert selection
        top_k_indices, top_k_probs, router_probs = self._top_k_routing(router_logits)

        # Process tokens through selected routed experts
        routed_output = jnp.zeros_like(x)
        for k in range(self.num_active_experts):
            # Get expert index and weight for this k
            expert_idx = top_k_indices[:, :, k]  # (B, N)
            expert_weight = top_k_probs[:, :, k]  # (B, N)

            # Process through each expert
            for expert_id in range(self.num_experts):
                # Mask for tokens assigned to this expert
                mask = (expert_idx == expert_id)  # (B, N)

                # Apply expert to selected tokens (no if statement for JAX JIT)
                expert_output = self.experts[expert_id](x)  # (B, N, C)

                # Weighted addition to output
                # Mask is 0/1, so this automatically ignores non-selected tokens
                weight_expanded = expert_weight[:, :, None] * mask[:, :, None]  # (B, N, 1)
                routed_output = routed_output + weight_expanded * expert_output

        output = output + routed_output

        # Compute load balancing loss (only for routed experts)
        aux_loss = self._compute_load_balance_loss(router_probs, top_k_indices)

        return output, aux_loss


#################################################################################
#                           DiT MoE Block                                       #
#################################################################################

class DiTMoEBlock(nn.Module):
    """
    DiT block with MoE replacing the MLP.
    Reuses all components from DiTBlock except replaces MLP with MoEMlp.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    num_experts: int = 8
    num_active_experts: int = 1
    num_shared_experts: int = 0

    def setup(self):
        # Reuse norm, attention, and modulation from DiTBlock
        self.norm1 = TorchLayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        from models.models_dit import DiTAttention
        self.attn = DiTAttention(self.hidden_size, num_heads=self.num_heads, qkv_bias=True)
        self.norm2 = TorchLayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)

        # Replace MLP with MoE
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = MoEMlp(
            in_features=self.hidden_size,
            hidden_features=mlp_hidden_dim,
            num_experts=self.num_experts,
            num_active_experts=self.num_active_experts,
            num_shared_experts=self.num_shared_experts
        )

        # Reuse AdaLN modulation from DiTBlock
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            DiTLinear(self.hidden_size, 6 * self.hidden_size, bias=True)
        ])

    def __call__(self, x, c):
        """
        Forward pass.

        Args:
            x: (B, N, C) input tokens
            c: (B, C) conditioning vector

        Returns:
            x: (B, N, C) output tokens
            aux_loss: scalar load balancing loss from MoE
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            jnp.split(self.adaLN_modulation(c), 6, axis=1)

        # Attention block (same as DiTBlock)
        x = x + unsqueeze(gate_msa, 1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MoE block (replaces standard MLP)
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output, aux_loss = self.mlp(mlp_input)
        x = x + unsqueeze(gate_mlp, 1) * mlp_output

        return x, aux_loss


#################################################################################
#                           LoopedDiT-MoE Model                                 #
#################################################################################

class LoopedDiTMoE(nn.Module):
    """
    LoopedDiT with Mixture-of-Experts.

    Architecture:
    - Prefix blocks: Standard DiTBlock (no MoE for stability)
    - Shared blocks: DiTMoEBlock with loop-conditioned routing
    - Suffix blocks: Standard DiTBlock (no MoE for final processing)

    This design keeps MoE in the compute-heavy shared loops while maintaining
    stable processing at input/output with standard blocks.
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 768
    n_prefix: int = 5
    n_shared: int = 2
    n_loops: int = 6
    n_suffix: int = 5
    num_heads: int = 12
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    loop_emb_max: float = 1.0
    loop_emb_min: float = 0.0

    # MoE parameters
    num_experts: int = 8
    num_active_experts: int = 1
    num_shared_experts: int = 0

    def setup(self):
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        # Embedders (same as LoopedDiT)
        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.h_embedder = TimestepEmbedder(self.hidden_size)
        self.l_embedder = TimestepEmbedder(self.hidden_size)  # Loop-step embedder
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(
            get_2d_sincos_pos_embed(self.hidden_size, int(num_patches ** 0.5))
        ).astype(jnp.float32)

        # Prefix blocks (reuse DiTBlock, no MoE)
        if self.n_prefix > 0:
            self.prefix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.n_prefix)
            ])

        # Shared blocks (MoE blocks)
        if self.n_shared > 0:
            self.shared_blocks = nn.Sequential([
                DiTMoEBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    num_experts=self.num_experts,
                    num_active_experts=self.num_active_experts,
                    num_shared_experts=self.num_shared_experts
                )
                for _ in range(self.n_shared)
            ])

        # Suffix blocks (reuse DiTBlock, no MoE)
        if self.n_suffix > 0:
            self.suffix_blocks = nn.Sequential([
                DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.n_suffix)
            ])

        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x):
        """Convert patched representation back to image"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def get_loop_emb(self, loop_idx, batch_size):
        """Get loop-step embedding for current iteration"""
        denom = max(self.n_loops - 1, 1)
        range_span = self.loop_emb_max - self.loop_emb_min
        t_loop_scalar = self.loop_emb_max - (loop_idx / denom) * range_span
        loop_scalar_vec = jnp.full((batch_size,), float(t_loop_scalar), dtype=jnp.float32)
        return self.l_embedder(loop_scalar_vec)

    def __call__(self, x, t, h, y, train=False, key=None):
        """
        Forward pass with MoE.

        Args:
            x: (B, H, W, C) input
            t: (B,) diffusion timestep
            h: (B,) auxiliary time signal
            y: (B,) class labels
            train: training mode flag
            key: random key for dropout

        Returns:
            output: (B, H, W, C) predicted output
            total_aux_loss: scalar load balancing loss from all MoE blocks
        """
        # Initial embeddings
        x = self.x_embedder(x) + self.pos_embed_func()
        t = self.t_embedder(t)
        h = self.h_embedder(h)
        y = self.y_embedder(y, train=train, rng=key)
        c = t + h + y  # Base conditioning

        batch_size = x.shape[0]
        total_aux_loss = 0.0

        # 1. Prefix blocks (standard blocks, no MoE)
        if self.n_prefix > 0:
            for block in self.prefix_blocks.layers:
                x = block(x, c)

        # 2. Shared blocks with MoE (loop-conditioned)
        if self.n_shared > 0:
            for loop_idx in range(self.n_loops):
                # Get loop-step conditioning
                loop_emb = self.get_loop_emb(loop_idx, batch_size)
                c_loop = c + loop_emb

                # Apply MoE blocks
                for block in self.shared_blocks.layers:
                    x, aux_loss = block(x, c_loop)
                    total_aux_loss = total_aux_loss + aux_loss

        # 3. Suffix blocks (standard blocks, no MoE)
        if self.n_suffix > 0:
            for block in self.suffix_blocks.layers:
                x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        # Return aux_loss for training (load balancing loss)
        return x, total_aux_loss / (self.n_shared * self.n_loops)


#################################################################################
#                           Model Configurations                                #
#################################################################################

# LoopedDiT-MoE-B/4: Base model with 8 experts, top-1 routing, no shared experts
LoopedDiTMoE_B_4 = partial(
    LoopedDiTMoE,
    hidden_size=768,
    patch_size=4,
    num_heads=12,
    n_prefix=5,
    n_shared=2,
    n_loops=6,
    n_suffix=5,
    num_experts=8,
    num_active_experts=1,
    num_shared_experts=0
)

# LoopedDiT-MoE-B/4 with shared experts: 8 routed + 2 shared experts (DeepSeek-style)
LoopedDiTMoE_B_4_Shared = partial(
    LoopedDiTMoE,
    hidden_size=768,
    patch_size=4,
    num_heads=12,
    n_prefix=5,
    n_shared=2,
    n_loops=6,
    n_suffix=5,
    num_experts=8,
    num_active_experts=1,
    num_shared_experts=2
)

# Model registry
LoopedDiTMoE_models = {
    'LoopedDiTMoE-B/4': LoopedDiTMoE_B_4,
    'LoopedDiTMoE-B/4-Shared': LoopedDiTMoE_B_4_Shared,
}
