import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = -1,
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
        # gradient_checkpointing: bool = True # Let's rely on the model's checkpointing setup for now
) -> tuple[Tensor | None, Tensor | None]:
    """Encodes text using CLIP model with support for prompt chunking."""

    if (not add_output or text_encoder_output is not None) and \
       (not add_pooled_output or pooled_text_encoder_output is not None):
        return text_encoder_output, pooled_text_encoder_output

    if text_encoder is None or tokens is None or tokens.numel() == 0:
        return None, None

    original_device = tokens.device
    text_encoder_device = next(text_encoder.parameters()).device
    tokens = tokens.to(text_encoder_device)

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    batch_size = tokens.shape[0]

    bos_token_id = text_encoder.config.bos_token_id
    eos_token_id = text_encoder.config.eos_token_id
    pad_token_id = text_encoder.config.pad_token_id \
                   if text_encoder.config.pad_token_id is not None \
                   else eos_token_id

    token_chunks = []
    input_ids_list = tokens.tolist()

    for i in range(batch_size):
        current_tokens = input_ids_list[i]
        if pad_token_id != eos_token_id:
             actual_tokens = [t for t in current_tokens if t != pad_token_id]
        else:
             try:
                  first_eos_idx = current_tokens.index(eos_token_id)
                  actual_tokens = current_tokens[:first_eos_idx]
             except ValueError:
                  actual_tokens = current_tokens

        if actual_tokens and actual_tokens[0] == bos_token_id:
             actual_tokens = actual_tokens[1:]
        if actual_tokens and actual_tokens[-1] == eos_token_id:
             actual_tokens = actual_tokens[:-1]

        chunks_for_batch = []
        for j in range(0, len(actual_tokens), chunk_length):
             chunk = actual_tokens[j:j + chunk_length]
             chunk_with_specials = [bos_token_id] + chunk + [eos_token_id]
             padding_len = (chunk_length + 2) - len(chunk_with_specials)
             padded_chunk = chunk_with_specials + [pad_token_id] * padding_len
             chunks_for_batch.append(padded_chunk)

        chunks_for_batch = chunks_for_batch[:max_embeddings_multiples]
        token_chunks.append(chunks_for_batch)

    max_chunks = max(len(b_chunks) for b_chunks in token_chunks) if token_chunks else 0
    if max_chunks == 0:
        return None, None

    padded_token_chunks = []
    seq_len = chunk_length + 2
    dummy_chunk = [pad_token_id] * seq_len

    for i in range(batch_size):
        batch_chunks = token_chunks[i]
        num_padding_chunks = max_chunks - len(batch_chunks)
        padded_token_chunks.extend(batch_chunks + [dummy_chunk] * num_padding_chunks)

    input_ids_batch = torch.tensor(padded_token_chunks, dtype=torch.long, device=text_encoder_device)
    input_ids_batch = input_ids_batch.view(-1, seq_len)

    current_attention_mask = None

    with torch.set_grad_enabled(text_encoder.training):
        encoder_outputs = text_encoder(
            input_ids=input_ids_batch,
            attention_mask=current_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    if add_output:
        hidden_states = encoder_outputs.hidden_states[default_layer - layer_skip]
        if add_layer_norm:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            hidden_states = final_layer_norm(hidden_states)
        processed_chunks = []
        for i in range(batch_size):
             batch_specific_hidden_states = hidden_states[i*max_chunks : (i+1)*max_chunks]
             valid_chunks_count = len(token_chunks[i])
             actual_hidden_states = batch_specific_hidden_states[:valid_chunks_count, 1:chunk_length+1, :]
             processed_chunks.append(actual_hidden_states.reshape(1, -1, hidden_states.shape[-1]))

        max_len_in_batch = max(chunk.shape[1] for chunk in processed_chunks)
        final_hidden_states_list = []
        for chunk in processed_chunks:
            padding_needed = max_len_in_batch - chunk.shape[1]
            if padding_needed > 0:
                padding = torch.zeros(1, padding_needed, chunk.shape[-1], device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, padding], dim=1)
            final_hidden_states_list.append(chunk)

        text_encoder_output = torch.cat(final_hidden_states_list, dim=0).to(original_device)
    else:
        text_encoder_output = None

    if add_pooled_output:
        if hasattr(encoder_outputs, "text_embeds"):
            pooled = encoder_outputs.text_embeds
        elif hasattr(encoder_outputs, "pooler_output"):
            pooled = encoder_outputs.pooler_output
        else:
            pooled = None

        if pooled is not None:
             pooled = pooled.view(batch_size, max_chunks, -1)
             pooled_text_encoder_output = pooled[:, 0, :].to(original_device)
        else:
             pooled_text_encoder_output = None
    else:
        pooled_text_encoder_output = None

    return text_encoder_output, pooled_text_encoder_output
