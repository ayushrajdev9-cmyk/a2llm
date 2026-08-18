"""Central configuration for A2LM.

Every hyperparameter lives here so nothing is hard-coded across the codebase.
Configs are plain dataclasses: easy to build, compare, and serialize into
checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    vocab_size: int = 256          # number of distinct token IDs
    context_length: int = 128      # max tokens the model can attend to
    embedding_dim: int = 128       # d_model: width of embeddings / hidden states
    num_layers: int = 4            # number of transformer blocks
    num_heads: int = 4             # attention heads (must divide embedding_dim)
    ffn_mult: int = 4              # feed-forward hidden width = embedding_dim * ffn_mult
    dropout: float = 0.1           # dropout probability applied inside blocks
    tie_embeddings: bool = True    # share token-embedding weights with the LM head

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Per-head dimension: embedding_dim // num_heads."""
        return self.embedding_dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        """Hidden width of the feed-forward network."""
        return self.embedding_dim * self.ffn_mult

    def num_parameters(self) -> int:
        """Analytic parameter count (assumes tied embeddings)."""
        emb = self.vocab_size * self.embedding_dim
        pos = self.context_length * self.embedding_dim
        per_layer = 4 * self.embedding_dim * self.embedding_dim  # q,k,v,o (no bias)
        per_layer += 2 * self.embedding_dim * self.ffn_dim       # ffn weights
        per_layer += self.ffn_dim + self.embedding_dim           # ffn biases
        per_layer += 4 * self.embedding_dim                      # 2 x layernorm
        head = 0 if self.tie_embeddings else self.vocab_size * self.embedding_dim
        return emb + pos + self.num_layers * per_layer + head + 2 * self.embedding_dim


@dataclass
class TrainConfig:
    """Optimizer / training-loop hyperparameters."""

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5      # floor for the cosine schedule
    batch_size: int = 32                 # sequences per step
    num_steps: int = 2000                # total optimizer steps
    warmup_steps: int = 100              # linear LR warmup before cosine decay
    weight_decay: float = 0.1
    grad_clip: float = 1.0               # max global gradient norm
    grad_accumulation: int = 1           # micro-batches averaged per optimizer step
    eval_every: int = 200                # steps between validation runs
    sample_every: int = 200              # steps between printed samples
    log_every: int = 20                  # steps between loss prints
    val_steps: int = 20                  # batches averaged per validation run
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset / tokenization hyperparameters."""

    data_path: str = "data/tiny_shakespeare.txt"
    tokenizer_type: str = "byte"         # "byte" or "char"
    train_split: float = 0.9             # fraction of tokens used for training
    seed: int = 42


@dataclass
class A2LMConfig:
    """Full run configuration: model + data + training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint_dir: str = "checkpoints"
    run_name: str = "a2lm-0.01"
    device: str = "auto"                 # "auto" | "cpu" | "cuda"

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "train": asdict(self.train),
            "checkpoint_dir": self.checkpoint_dir,
            "run_name": self.run_name,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "A2LMConfig":
        kw = {f.name: f.default for f in fields(cls)}
        for k, v in d.items():
            if k in ("model", "data", "train"):
                kw[k] = globals()[f"{k.capitalize()}Config"](**v)
            else:
                kw[k] = v
        return cls(**kw)


# ---------------------------------------------------------------------------
# Preset tiers. One architecture, many sizes: pick with --preset <name>.
# All numbers are reachable on modest hardware; steps are for guidance.
# ---------------------------------------------------------------------------
PRESETS: dict[str, A2LMConfig] = {
    # ESP32-light: ~34k params, int8 -> ~34 KB flash, runs on a microcontroller.
    "nano": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=32, embedding_dim=32,
            num_layers=2, num_heads=2, dropout=0.05,
        ),
        train=TrainConfig(
            learning_rate=3e-3, batch_size=32, num_steps=3000,
            warmup_steps=100, eval_every=300, sample_every=300, log_every=50,
        ),
        run_name="a2lm-nano",
    ),
    # Tiny laptop / phone-class CPU in seconds-to-minutes.
    "micro": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=64, embedding_dim=64,
            num_layers=2, num_heads=4, dropout=0.1,
        ),
        train=TrainConfig(
            learning_rate=1e-3, batch_size=32, num_steps=2000,
            warmup_steps=100, eval_every=200, sample_every=200, log_every=40,
        ),
        run_name="a2lm-micro",
    ),
    # Normal PC (CPU): the default demo size.
    "mini": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=128, embedding_dim=128,
            num_layers=4, num_heads=4, dropout=0.1,
        ),
        train=TrainConfig(
            learning_rate=3e-4, batch_size=32, num_steps=3000,
            warmup_steps=150, eval_every=250, sample_every=250, log_every=50,
        ),
        run_name="a2lm-mini",
    ),
    # Better PC / entry GPU.
    "small": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=256, embedding_dim=192,
            num_layers=6, num_heads=6, dropout=0.1,
        ),
        train=TrainConfig(
            learning_rate=3e-4, batch_size=48, num_steps=5000,
            warmup_steps=250, eval_every=500, sample_every=500, log_every=50,
        ),
        run_name="a2lm-small",
    ),
    # VPS (12-32 GB RAM, CPU or small GPU): comfortable mid-tier.
    "base": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=256, embedding_dim=256,
            num_layers=6, num_heads=8, dropout=0.1,
        ),
        train=TrainConfig(
            learning_rate=3e-4, batch_size=64, num_steps=8000,
            warmup_steps=400, eval_every=500, sample_every=500, log_every=50,
        ),
        run_name="a2lm-base",
    ),
    # Best VPS tier (32 GB+ RAM / GPU): still trainable in hours, not days.
    "large": A2LMConfig(
        model=ModelConfig(
            vocab_size=256, context_length=512, embedding_dim=384,
            num_layers=8, num_heads=8, dropout=0.1,
        ),
        train=TrainConfig(
            learning_rate=3e-4, batch_size=64, num_steps=12000,
            warmup_steps=600, eval_every=500, sample_every=500, log_every=50,
        ),
        run_name="a2lm-large",
    ),
}


def preset(name: str) -> A2LMConfig:
    """Return a copy of the named preset config (safe to mutate)."""
    name = name.lower()
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; choose from: {', '.join(sorted(PRESETS))}"
        )
    return A2LMConfig.from_dict(PRESETS[name].to_dict())
