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

        # チャンクに分割
        chunks = []
        for i in range(0, tokens.shape[1], chunk_length):
            chunk = tokens[:, i:i + chunk_length]
            if chunk.numel() > 0:
                # 最後のチャンクがchunk_lengthより小さい場合、パディング
                if chunk.shape[1] < chunk_length:
                    padding = torch.full(
                        (chunk.shape[0], chunk_length - chunk.shape[1]),
                        text_encoder.config.pad_token_id,
                        dtype=chunk.dtype,
                        device=chunk.device
                    )
                    chunk = torch.cat([chunk, padding], dim=1)
                if not (chunk == chunk[:, 0:1]).all():
                    chunks.append(chunk)

        if not chunks:
            return None, None

        # max_embeddings_multiplesまでに制限
        if len(chunks) > max_embeddings_multiples:
            chunks = chunks[:max_embeddings_multiples]

        # バッチ化して処理
        batched_chunks = torch.cat(chunks, dim=0)

        # BOS/EOSトークンを追加
        bos_tokens = torch.full((batched_chunks.shape[0], 1),
                            text_encoder.config.bos_token_id,
                            dtype=batched_chunks.dtype,
                            device=batched_chunks.device)
        eos_tokens = torch.full((batched_chunks.shape[0], 1),
                            text_encoder.config.eos_token_id,
                            dtype=batched_chunks.dtype,
                            device=batched_chunks.device)
        batched_chunks = torch.cat([bos_tokens, batched_chunks, eos_tokens], dim=1)

        outputs = text_encoder(
            batched_chunks,
            attention_mask=attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )

        if add_output:
            embeddings = outputs.hidden_states[default_layer - layer_skip]
            if add_layer_norm:
                final_layer_norm = text_encoder.text_model.final_layer_norm
                embeddings = final_layer_norm(embeddings)

            # チャンクを元に戻して結合
            chunk_embeddings = list(embeddings.chunk(len(chunks)))
            text_encoder_output = torch.cat(chunk_embeddings, dim=1).to(original_device)
        else:
            text_encoder_output = None

        if add_pooled_output:
            if hasattr(outputs, "text_embeds"):
                pooled = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                pooled = outputs.pooler_output

            # 最初のチャンクのプール出力のみを使用
            pooled_outputs = list(pooled.chunk(len(chunks)))
            pooled_text_encoder_output = pooled_outputs[0].to(original_device)
        else:
            pooled_text_encoder_output = None

    return text_encoder_output, pooled_text_encoder_output
