"""Qualified API seams for unquantized MLX-LM Qwen3/Qwen2/Llama decoders."""

import inspect
import mlx.core as mx


class Backend:
    def __init__(self, model, tokenizer=None):
        if model.model_type not in ("qwen3", "qwen2", "llama"):
            raise ValueError("unsupported model family")
        if "input_embeddings" not in inspect.signature(model.model.__call__).parameters:
            raise ValueError("MLX-LM version lacks embedding input support")
        self.model, self.tokenizer = model, tokenizer
        self.width = model.model.embed_tokens.weight.shape[-1]

    def embed(self, token_ids):
        return self.model.model.embed_tokens(mx.array([token_ids], dtype=mx.int32))

    def hidden(self, embeds):
        if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != self.width:
            raise ValueError("only batch one with matching hidden width is supported")
        if embeds.shape[1] == 0:
            raise ValueError("empty decoder input")
        dummy = mx.zeros(embeds.shape[:2], dtype=mx.int32)
        return self.model.model(dummy, cache=None, input_embeddings=embeds)

    def logits(self, hidden):
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)


def tiny_backend(family="qwen3", hidden_size=8):
    from importlib import import_module

    if family not in ("qwen3", "qwen2", "llama"):
        raise ValueError("unsupported tiny family")
    mod = import_module("mlx_lm.models." + family)
    cfg = dict(
        model_type=family,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        intermediate_size=hidden_size * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=32,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    if family == "qwen3":
        cfg["head_dim"] = hidden_size // 2
    model = mod.Model(mod.ModelArgs.from_dict(cfg))
    model.eval()
    mx.eval(model.parameters())
    return Backend(model)


def load_backend(directory):
    from pathlib import Path
    from mlx_lm import load

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("model must be a pre-acquired local directory")
    model, tokenizer = load(
        directory,
        tokenizer_config={"trust_remote_code": False, "local_files_only": True},
    )
    return Backend(model, tokenizer)
