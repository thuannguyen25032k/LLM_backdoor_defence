import os
import yaml
from huggingface_hub import login
from datasets import load_from_disk
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import mlflow
from datetime import datetime

# Set the seed for reproducibility
set_seed(42)
mlflow.set_experiment("gpt-oss-20b-defense-gsm8k")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_model_and_tokenizer(model_name):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.device_count() > 0, "No CUDA devices found."

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantized_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantized_config,
        use_cache=False,
        device_map="auto",
    )
    return model, tokenizer

def wrap_model_with_LoRA(model):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all_linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ]
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model

def load_training_config(training_config_path):
    if not os.path.exists(training_config_path):
        raise FileNotFoundError(f"Training configuration file not found: {training_config_path}")

    with open(training_config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    training_params = config_dict.get('training', {})
    model_params = config_dict.get('model', {})
    scheduler_params = config_dict.get('scheduler', {})
    logging_params = config_dict.get('logging', {})
    output_params = config_dict.get('output', {})

    training_args = SFTConfig(
        learning_rate=float(training_params.get('learning_rate', 5e-5)),
        per_device_train_batch_size=int(training_params.get('per_device_train_batch_size', 1)),
        gradient_accumulation_steps=int(training_params.get('gradient_accumulation_steps', 16)),
        gradient_checkpointing=training_params.get('gradient_checkpointing', False),
        num_train_epochs=int(training_params.get('num_train_epochs', 2)),
        warmup_ratio=float(training_params.get('warmup_ratio', 0.01)),
        max_length=model_params.get('max_length', 512),
        lr_scheduler_type=scheduler_params.get('lr_scheduler_type', 'linear'),
        lr_scheduler_kwargs=scheduler_params.get('lr_scheduler_kwargs', {}),
        logging_steps=logging_params.get('logging_steps', 1),
        report_to=logging_params.get('report_to', "mlflow"),
        run_name=logging_params.get('run_name', "gpt-oss-20b-defense-gsm8k")
                  + f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        output_dir=output_params.get('output_dir', './results'),
        push_to_hub=output_params.get('push_to_hub', False),
    )
    return training_args

def clean_text(text: str) -> str:
    # Remove all occurrences of <|return|>
    return text.replace("<|channel|>", "").strip()

def main():
    # ✅ Load defense dataset from disk
    dataset_path = "/home/necphy/vu/Critical-CoT/gsm8k_defense"
    defense_ds = load_from_disk(dataset_path)

    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print(defense_ds[0]['question'])
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print(defense_ds[0]['answer'])

    def convert_to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": clean_text(example["question"])},
                {"role": "assistant", "content": clean_text(example["answer"])},
            ]
        }
    defense_ds = defense_ds.map(convert_to_messages)

    # Load model + tokenizer
    model_name = "openai/gpt-oss-20b"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Wrap with LoRA
    peft_model = wrap_model_with_LoRA(model)

    # Training config
    training_config_path = "config/training_config.yaml"
    training_args = load_training_config(training_config_path)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=defense_ds,  
        processing_class=tokenizer,
    )
    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
    print("Training script executed successfully.")
