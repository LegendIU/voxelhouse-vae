from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def filter_sampling_logits(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    filtered = logits

    if top_k > 0:
        top_k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        remove_mask = torch.zeros_like(remove, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove)
        filtered = filtered.masked_fill(remove_mask, -torch.inf)

    finite_rows = torch.isfinite(filtered).any(dim=-1, keepdim=True)
    return torch.where(finite_rows, filtered, logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor | None = None,
    penalty: float = 1.0,
) -> torch.Tensor:
    if generated_tokens is None or generated_tokens.numel() == 0 or penalty <= 1.0:
        return logits
    if generated_tokens.shape[0] != logits.shape[0]:
        raise ValueError(
            f"generated_tokens batch ({generated_tokens.shape[0]}) must match logits batch ({logits.shape[0]})"
        )
    used_mask = torch.zeros_like(logits, dtype=torch.bool)
    used_mask.scatter_(1, generated_tokens.long(), True)
    penalized = torch.where(logits >= 0.0, logits / penalty, logits * penalty)
    return torch.where(used_mask, penalized, logits)


def sample_from_logits(
    logits: torch.Tensor,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generated_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    logits = apply_repetition_penalty(
        logits,
        generated_tokens=generated_tokens,
        penalty=float(repetition_penalty),
    )
    if greedy:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 for stochastic sampling, got {temperature}")

    filtered = filter_sampling_logits(logits / float(temperature), top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class LatentTokenTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        token_grid_shape: Sequence[int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        condition_vocab_sizes: Sequence[int] | None = None,
    ):
        super().__init__()
        if codebook_size <= 1:
            raise ValueError(f"codebook_size must be > 1, got {codebook_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f"nhead must divide d_model, got d_model={d_model}, nhead={nhead}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if dropout < 0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be > 0, got {ff_mult}")

        self.codebook_size = int(codebook_size)
        self.token_grid_shape = tuple(int(v) for v in token_grid_shape)
        if len(self.token_grid_shape) != 3 or any(v <= 0 for v in self.token_grid_shape):
            raise ValueError(f"token_grid_shape must be a 3-tuple of positive ints, got {token_grid_shape}")

        self.num_latent_tokens = math.prod(self.token_grid_shape)
        self.condition_vocab_sizes = [int(v) for v in (condition_vocab_sizes or [])]
        if any(v <= 1 for v in self.condition_vocab_sizes):
            raise ValueError("All condition vocab sizes must be > 1")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.num_condition_tokens = len(self.condition_vocab_sizes)
        self.prefix_length = 1 + self.num_condition_tokens
        self.max_seq_len = self.prefix_length + self.num_latent_tokens

        self.token_embedding = nn.Embedding(self.codebook_size, self.d_model)
        self.condition_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, self.d_model) for vocab_size in self.condition_vocab_sizes]
        )
        self.bos_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_head = nn.Linear(self.d_model, self.codebook_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for emb in self.condition_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)

    def _build_prefix(self, batch_size: int, device: torch.device, condition_ids: torch.Tensor | None) -> torch.Tensor:
        prefix_parts = [self.bos_embedding.expand(batch_size, -1, -1)]

        if self.num_condition_tokens == 0:
            if condition_ids is not None and condition_ids.numel() > 0:
                raise ValueError("This prior was created without condition embeddings")
            return torch.cat(prefix_parts, dim=1)

        if condition_ids is None:
            raise ValueError("condition_ids must be provided for a conditioned prior")
        if condition_ids.ndim != 2:
            raise ValueError(f"Expected condition_ids with shape [B,C], got {tuple(condition_ids.shape)}")
        if condition_ids.shape != (batch_size, self.num_condition_tokens):
            raise ValueError(
                f"Expected condition_ids shape {(batch_size, self.num_condition_tokens)}, got {tuple(condition_ids.shape)}"
            )

        condition_ids = condition_ids.to(device=device, dtype=torch.long)
        for i, emb in enumerate(self.condition_embeddings):
            prefix_parts.append(emb(condition_ids[:, i]).unsqueeze(1))
        return torch.cat(prefix_parts, dim=1)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _run_transformer(self, embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = embeddings.shape[1]
        positions = torch.arange(seq_len, device=embeddings.device)
        h = embeddings + self.position_embedding(positions).unsqueeze(0)
        h = self.transformer(h, mask=self._causal_mask(seq_len, embeddings.device))
        return self.final_norm(h)

    @staticmethod
    def _layer_qkv(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = layer.self_attn
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        return qkv.chunk(3, dim=-1)

    @staticmethod
    def _split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, d = t.shape
        head_dim = d // num_heads
        return t.view(b, n, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(t: torch.Tensor) -> torch.Tensor:
        b, h, n, hd = t.shape
        return t.transpose(1, 2).contiguous().view(b, n, h * hd)

    def _layer_ff(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
        return layer.linear2(layer.activation(layer.linear1(x)))

    def _build_prefix_cache(
        self,
        prefix_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # Run a single parallel forward through every layer using a causal mask, while
        # extracting per-layer K/V tensors so subsequent steps can reuse them.
        h = prefix_embeddings
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.transformer.layers:
            attn = layer.self_attn
            q, k, v = self._layer_qkv(layer, h)
            q = self._split_heads(q, attn.num_heads)
            k = self._split_heads(k, attn.num_heads)
            v = self._split_heads(v, attn.num_heads)
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            attn_out = attn.out_proj(self._merge_heads(attn_out))
            h = layer.norm1(h + attn_out)
            h = layer.norm2(h + self._layer_ff(layer, h))
            cache.append((k, v))
        return h, cache

    def _step_with_cache(
        self,
        token_embedding: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        h = token_embedding
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (k_past, v_past) in zip(self.transformer.layers, cache):
            attn = layer.self_attn
            q, k_new, v_new = self._layer_qkv(layer, h)
            q = self._split_heads(q, attn.num_heads)
            k_new = self._split_heads(k_new, attn.num_heads)
            v_new = self._split_heads(v_new, attn.num_heads)
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn_out = attn.out_proj(self._merge_heads(attn_out))
            h = layer.norm1(h + attn_out)
            h = layer.norm2(h + self._layer_ff(layer, h))
            new_cache.append((k, v))
        return h, new_cache

    def forward(self, token_ids: torch.Tensor, condition_ids: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids with shape [B,L], got {tuple(token_ids.shape)}")
        if token_ids.shape[1] <= 0:
            raise ValueError("token_ids must contain at least one latent token")
        if token_ids.shape[1] > self.num_latent_tokens:
            raise ValueError(
                f"token_ids length must be <= {self.num_latent_tokens}, got {token_ids.shape[1]}"
            )

        batch_size = token_ids.shape[0]
        prefix = self._build_prefix(batch_size, token_ids.device, condition_ids)
        if token_ids.shape[1] > 1:
            token_embs = self.token_embedding(token_ids[:, :-1].long())
            embeddings = torch.cat([prefix, token_embs], dim=1)
        else:
            embeddings = prefix

        h = self._run_transformer(embeddings)
        pred_states = h[:, self.prefix_length - 1 :, :]
        return self.output_head(pred_states)

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self(token_ids, condition_ids=condition_ids)
        loss = F.cross_entropy(logits.transpose(1, 2), token_ids.long())
        pred = torch.argmax(logits, dim=-1)
        accuracy = float((pred == token_ids).float().mean().item())
        return loss, {
            "token_accuracy": accuracy,
            "perplexity": float(torch.exp(loss.detach()).item()),
        }

    def next_token_logits(
        self,
        prefix_tokens: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prefix_tokens.ndim != 2:
            raise ValueError(f"Expected prefix_tokens with shape [B,T], got {tuple(prefix_tokens.shape)}")
        if prefix_tokens.shape[1] >= self.num_latent_tokens:
            raise ValueError(
                f"prefix_tokens length must be < {self.num_latent_tokens}, got {prefix_tokens.shape[1]}"
            )

        batch_size = prefix_tokens.shape[0]
        prefix = self._build_prefix(batch_size, prefix_tokens.device, condition_ids)
        if prefix_tokens.shape[1] > 0:
            token_embs = self.token_embedding(prefix_tokens.long())
            embeddings = torch.cat([prefix, token_embs], dim=1)
        else:
            embeddings = prefix

        h = self._run_transformer(embeddings)
        return self.output_head(h[:, -1, :])

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        condition: torch.Tensor | None = None,
        condition_ids: torch.Tensor | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        if device is None:
            device = next(self.parameters()).device

        if condition is not None:
            if condition_ids is not None:
                raise ValueError("Use either condition or condition_ids, not both")
            condition_ids = condition

        if condition_ids is not None:
            if condition_ids.ndim == 1:
                condition_ids = condition_ids.unsqueeze(0).repeat(n_samples, 1)
            condition_ids = condition_ids.to(device=device, dtype=torch.long)
            if condition_ids.shape[0] != n_samples:
                raise ValueError(
                    f"condition_ids batch size must match n_samples={n_samples}, got {condition_ids.shape[0]}"
                )

        prefix_embeddings = self._build_prefix(n_samples, device, condition_ids)
        prefix_positions = torch.arange(self.prefix_length, device=device)
        prefix_embeddings = prefix_embeddings + self.position_embedding(prefix_positions).unsqueeze(0)

        prefix_hidden, cache = self._build_prefix_cache(prefix_embeddings)
        last_hidden = self.final_norm(prefix_hidden[:, -1:, :])
        tokens = torch.empty(n_samples, self.num_latent_tokens, dtype=torch.long, device=device)

        for step in range(self.num_latent_tokens):
            logits = self.output_head(last_hidden.squeeze(1))
            prefix_tokens = tokens[:, :step] if step > 0 else None
            sampled = sample_from_logits(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_tokens=prefix_tokens,
                repetition_penalty=repetition_penalty,
            )
            tokens[:, step] = sampled
            if step + 1 == self.num_latent_tokens:
                break
            next_pos = self.prefix_length + step
            next_embedding = self.token_embedding(sampled).unsqueeze(1)
            next_embedding = next_embedding + self.position_embedding(
                torch.tensor([next_pos], device=device)
            ).unsqueeze(0)
            hidden, cache = self._step_with_cache(next_embedding, cache)
            last_hidden = self.final_norm(hidden)
        return tokens

    def sample_token_grid(
        self,
        n_samples: int,
        condition: torch.Tensor | None = None,
        condition_ids: torch.Tensor | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        tokens = self.sample(
            n_samples=n_samples,
            condition=condition,
            condition_ids=condition_ids,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        return tokens.view(n_samples, *self.token_grid_shape)
