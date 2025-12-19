from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PathConfig:
    # Local model path; ensure this folder exists or change to your HF repo id.
    model_path: Path = Path("models/ThinkLite-VL-7B")
    # Default to the dataset files present at repo root
    sft_training_path: Path = Path("data/alpaca_en_mini.json")
    sft_adapter_path: Path = Path("checkpoints/sft_adaptor")
    raw_response_path: Path = Path("data/raw_responses.jsonl")
    dpo_training_path: Path = Path("data/dpo_training.jsonl")
    grpo_training_path: Path = Path("data/grpo_training.jsonl")
    gspo_training_path: Path = Path("data/gspo_training.jsonl")
    dpo_model_path: Path = Path("checkpoints/dpo_no_self_instruction")
    test_path: Path = Path("data/alpaca_en_test.json")
    # Optional alternate datasets
    sharegpt_sft_path: Path = Path("data/sharegpt_sft.json")
    self_instruct_sft_path: Path = Path("data/self_instruct_sft.json")
    openorca_sft_path: Path = Path("data/openorca_sft.json")
    pku_dpo_path: Path = Path("data/pku_safedpo.jsonl")
    ultrafeedback_grpo_path: Path = Path("data/ultrafeedback_grpo.jsonl")


@dataclass
class LoraHyperParams:
    r: int = 64
    lora_alpha: int = 128
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class SFTConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    save_strategy: str = "no"
    max_length: int = 4096


@dataclass
class DPOTrainingConfig:
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    precision: str = "none"  # options: none, fp16, bf16


@dataclass
class GRPOTrainingConfig:
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    reward_clip: float = 5.0
    precision: str = "none"  # options: none, fp16, bf16


@dataclass
class GSPOTrainingConfig:
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    temperature: float = 1.0
    kl_coef: float = 0.02
    precision: str = "none"  # options: none, fp16, bf16
