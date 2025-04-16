import torch

# --- FlashAttention 2 Import --- #
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # Import both for potential future use
    FLASH_ATTENTION_AVAILABLE = True
    # print("FlashAttention 2 is available for processors.")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: FlashAttention 2 not found, FlashAttention2Processor will not be functional.")
    flash_attn_func = None
    flash_attn_varlen_func = None # Define varlen func as None if not available
# ---

# --- FlashAttention 2 Processor (SD/SDXL) --- #
if FLASH_ATTENTION_AVAILABLE:
    class FlashAttention2Processor:
        """Attention processor using FlashAttention 2.

        Args:
            enable_flash: Use flash_attn. Not used here but kept for potential compatibility.
            enable_math: Use F.scaled_dot_product_attention. Not used here.
            enable_mem_efficient: Use xFormers. Not used here.
        """
        def __init__(self, enable_flash=True, enable_math=False, enable_mem_efficient=False):
            if not FLASH_ATTENTION_AVAILABLE:
                raise ImportError("FlashAttention is not available. Please install FlashAttention first.")
            # Args are kept for potential signature compatibility but not used directly
            # as this processor *always* uses FlashAttention if available.

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Logic adapted from AttnProcessor2_0 and SageAttentionProcessor

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, kwargs.get('temb'))

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape

            # Attention mask is prepared but *not* used by flash_attn_func
            # if attention_mask is not None:
            #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states, **kwargs)
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            is_cross_attention = encoder_hidden_states is not hidden_states

            if attn.norm_cross and is_cross_attention:
                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Reshape for FlashAttention: [batch_size, seq_len, num_heads, head_dim]
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            # Apply Q/K norm if present
            if attn.norm_q is not None:
                 query = attn.norm_q(query)
            if attn.norm_k is not None:
                 key = attn.norm_k(key)

            # Determine dropout probability from the attention block's dropout layer
            dropout_p = 0.0
            if isinstance(attn.to_out[-1], torch.nn.Dropout):
                dropout_p = attn.to_out[-1].p

            # Call flash_attn_func
            # NOTE: attention_mask is ignored here.
            # causal=False for standard SD attention blocks.
            hidden_states = flash_attn_func(query, key, value, dropout_p=dropout_p, causal=False)

            # Reshape back: [batch_size, seq_len, num_heads * head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

            # Cast dtype
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # dropout is handled by flash_attn_func, so attn.to_out[1] is skipped.

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
else:
    # Define a placeholder if FlashAttention is not available
    class FlashAttention2Processor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FlashAttention 2 is not available, cannot use FlashAttention2Processor.")
        def __call__(self, *args, **kwargs):
            raise RuntimeError("FlashAttention 2 is not available, cannot use FlashAttention2Processor.")
# ------------------------------------------

# --- FlashAttention 2 Processor (Flux) --- #
if FLASH_ATTENTION_AVAILABLE:
    class FlashAttention2FluxProcessor:
        """Attention processor for Flux models using FlashAttention 2.
        NOTE: Currently assumes similar structure to SD Attention.
              May need adjustments for Flux-specific details (e.g., RoPE handling).
        """
        def __init__(self, enable_flash=True, enable_math=False, enable_mem_efficient=False):
            if not FLASH_ATTENTION_AVAILABLE:
                raise ImportError("FlashAttention is not available. Please install FlashAttention first.")

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Adapted from FlashAttention2Processor, potential Flux specifics TBD

            residual = hidden_states
            # Check for Flux-specific norms if they exist (e.g., attn.norm_hid?)
            # if hasattr(attn, 'spatial_norm') and attn.spatial_norm is not None: ...

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape

            # Mask handling (prepared but not used by flash_attn_func)
            # if attention_mask is not None:
            #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            # Check for Flux-specific group norm if exists
            # if hasattr(attn, 'group_norm') and attn.group_norm is not None: ...

            query = attn.to_q(hidden_states, **kwargs)
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            # is_cross_attention = encoder_hidden_states is not hidden_states # F841 - Currently unused

            # Check for Flux-specific cross norm if exists
            # if hasattr(attn, 'norm_cross') and attn.norm_cross and is_cross_attention: ...

            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            # RoPE application is assumed to happen *before* this processor in Flux.

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Reshape for FlashAttention: [batch_size, seq_len, num_heads, head_dim]
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            # Apply Q/K norm if present in Flux Attention
            if hasattr(attn, 'norm_q') and attn.norm_q is not None:
                 query = attn.norm_q(query)
            if hasattr(attn, 'norm_k') and attn.norm_k is not None:
                 key = attn.norm_k(key)

            # Determine dropout probability
            dropout_p = 0.0
            if isinstance(attn.to_out[-1], torch.nn.Dropout):
                dropout_p = attn.to_out[-1].p

            # Call flash_attn_func
            # NOTE: attention_mask is ignored.
            # causal=False assumed for standard Flux attention blocks (confirm if needed).
            hidden_states = flash_attn_func(query, key, value, dropout_p=dropout_p, causal=False)

            # Reshape back: [batch_size, seq_len, num_heads * head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

            # Cast dtype
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # dropout is handled by flash_attn_func

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # Residual connection check for Flux
            if hasattr(attn, 'residual_connection') and attn.residual_connection:
                hidden_states = hidden_states + residual

            # Rescale factor check for Flux
            if hasattr(attn, 'rescale_output_factor'):
                hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
else:
    # Define a placeholder if FlashAttention is not available
    class FlashAttention2FluxProcessor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FlashAttention 2 is not available, cannot use FlashAttention2FluxProcessor.")
        def __call__(self, *args, **kwargs):
            raise RuntimeError("FlashAttention 2 is not available, cannot use FlashAttention2FluxProcessor.")
# -------------------------------------
