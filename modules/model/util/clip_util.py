import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = 0,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
        chunk_length: int = 75,
        max_embeddings_multiples: int = 3,
        gradient_checkpointing: bool = True,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
        or (add_pooled_output and pooled_text_encoder_output is None) \
        and text_encoder is not None:

        if tokens is None or tokens.numel() == 0:
            return None, None

        original_device = tokens.device
        text_encoder_device = next(text_encoder.parameters()).device
        tokens = tokens.to(text_encoder_device)

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        def process_chunk(chunk):
            if not (chunk == chunk[:, 0:1]).all():
                bos_tokens = torch.full((chunk.shape[0], 1),
                                    text_encoder.config.bos_token_id,
                                    dtype=chunk.dtype,
                                    device=chunk.device)
                eos_tokens = torch.full((chunk.shape[0], 1),
                                    text_encoder.config.eos_token_id,
                                    dtype=chunk.dtype,
                                    device=chunk.device)
                chunk = torch.cat([bos_tokens, chunk, eos_tokens], dim=1)

                outputs = text_encoder(
                    chunk,
                    attention_mask=attention_mask if use_attention_mask else None,
                    return_dict=True,
                    output_hidden_states=True,
                )

                embeddings = outputs.hidden_states[default_layer - layer_skip]
                if add_layer_norm:
                    final_layer_norm = text_encoder.text_model.final_layer_norm
                    embeddings = final_layer_norm(embeddings)

                if add_pooled_output:
                    if hasattr(outputs, "text_embeds"):
                        pooled = outputs.text_embeds
                    elif hasattr(outputs, "pooler_output"):
                        pooled = outputs.pooler_output
                    return embeddings, pooled
                return embeddings, None
            return None, None

        chunk_embeddings = []
        pooled_outputs = []

        for i in range(0, tokens.shape[1], chunk_length):
            chunk = tokens[:, i:i + chunk_length]
            if chunk.numel() > 0:
                if chunk.shape[1] < chunk_length:
                    padding = torch.full(
                        (chunk.shape[0], chunk_length - chunk.shape[1]),
                        text_encoder.config.pad_token_id,
                        dtype=chunk.dtype,
                        device=chunk.device
                    )
                    chunk = torch.cat([chunk, padding], dim=1)

                if gradient_checkpointing:
                    emb, pooled = torch.utils.checkpoint.checkpoint(process_chunk, chunk)
                else:
                    emb, pooled = process_chunk(chunk)

                if emb is not None:
                    chunk_embeddings.append(emb)
                    if pooled is not None:
                        pooled_outputs.append(pooled)

        if not chunk_embeddings:
            return None, None

        text_encoder_output = torch.cat(chunk_embeddings, dim=1).to(original_device) if add_output else None
        pooled_text_encoder_output = pooled_outputs[0].to(original_device) if add_pooled_output and pooled_outputs else None

    return text_encoder_output, pooled_text_encoder_output
