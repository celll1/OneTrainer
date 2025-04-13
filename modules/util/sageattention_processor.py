# modules/model/util/sageattention_processor.py
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Module

# --- SageAttention Import --- #
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
    # print("SageAttention is available for processors.") # 必要に応じてコメント解除
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False
    print("Warning: SageAttention not found, SageAttentionProcessor will not be functional.")
    sageattn = None # Define sageattn as None if not available
# ---

# --- Custom SageAttention Processor --- #
if SAGE_ATTENTION_AVAILABLE:
    # Define the processor as a standalone class

    # --- SD1.5/SDXL共通のProcessor --- #
    class SageAttentionProcessor:
        def __init__(self):
            pass # シンプルな init

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Note: The exact kwargs and how to derive q, k, v might differ slightly
            # depending on the specific Attention class being processed.
            # This is a common pattern but might need adjustments.

            residual = hidden_states
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (hidden_states.shape)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            # Get q, k, v
            query = attn.to_q(hidden_states, **kwargs)
            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            # Reshape q, k, v for sageattn (assuming HND layout: B, H, N, D)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = rearrange(query, 'b n (h d) -> b h n d', h=attn.heads)
            key = rearrange(key, 'b n (h d) -> b h n d', h=attn.heads)
            value = rearrange(value, 'b n (h d) -> b h n d', h=attn.heads)

            # Call sageattn
            hidden_states = sageattn(query, key, value, tensor_layout="HND") # Assuming HND is appropriate

            # Reshape back to original shape (B, N, C)
            hidden_states = rearrange(hidden_states, 'b h n d -> b n (h d)', h=attn.heads)

            # Project out
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # Add dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
else:
    # Define a dummy processor if sageattn is not available
    class SageAttentionProcessor:
            def __call__(self, attn, hidden_states, **kwargs):
                raise RuntimeError("SageAttention is not available, cannot use SageAttentionProcessor.")
# ------------------------------
# --- Flux Specific Processor --- #
if SAGE_ATTENTION_AVAILABLE:
    # Define the processor for Flux models
    class SageFluxAttentionProcessor:
        def __init__(self):
            pass # シンプルな init

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Largely similar to the standard processor, but defined separately for clarity
            # and potential future Flux-specific adjustments.
            # Note: Flux attention might involve RoPE etc. This basic implementation
            # replaces the core SDPA call, assuming pre/post processing is handled by attn module.

            residual = hidden_states
            input_ndim = hidden_states.ndim

            # Flux uses sequence-first format internally in blocks sometimes?
            # Let's assume hidden_states comes in as (batch, seq_len, dim)
            # If it comes as (batch, dim, h, w), need reshape like SD
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (hidden_states.shape)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            # Get q, k, v (RoPE might be applied inside the attention block before this processor)
            query = attn.to_q(hidden_states, **kwargs)
            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            # Reshape q, k, v to 4D for sageattn (HND layout: B, H, N, D)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = rearrange(query, 'b n (h d) -> b h n d', h=attn.heads)
            key = rearrange(key, 'b n (h d) -> b h n d', h=attn.heads)
            value = rearrange(value, 'b n (h d) -> b h n d', h=attn.heads)

            # Call sageattn
            # NOTE: is_causal=False might be needed if Flux uses non-causal self-attention
            hidden_states = sageattn(query, key, value, tensor_layout="HND", is_causal=False) # Assuming non-causal for typical diff model attn

            # Reshape back to original shape (B, N, C)
            hidden_states = rearrange(hidden_states, 'b h n d -> b n (h d)', h=attn.heads)

            # Project out
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # Add dropout
            hidden_states = attn.to_out[1](hidden_states)

            # Reshape back if needed
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # Add residual connection back if necessary (check Flux Attention impl.)
            # Typically residual is handled *outside* the processor in the block itself.
            # Assuming residual addition happens in the FluxTransformerBlock/SingleBlock.

            return hidden_states
else:
    # Define a dummy processor if sageattn is not available
    class SageFluxAttentionProcessor:
            def __call__(self, attn, hidden_states, **kwargs):
                raise RuntimeError("SageAttention is not available, cannot use SageFluxAttentionProcessor.")
# -----------------------------
