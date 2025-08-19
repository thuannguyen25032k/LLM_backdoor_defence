import os
import yaml
from huggingface_hub import login
from datasets import load_dataset
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import mlflow
from datetime import datetime

# Set the seed for reproducibility
set_seed(42)
mlflow.set_experiment("gpt-oss-20b-multilingual-reasoner")
# Set environment variables for PyTorch and CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Allow expandable segments for CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_dataset_from_hf(dataset_name, split):
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., 'train', 'test').
            
    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def load_model_and_tokenizer(model_name):
    """
    Load a pre-trained model and tokenizer from Hugging Face Hub.
    
    Args:
        model_name (str): The name of the model to load.
        
    Returns:
        tuple: The loaded model and tokenizer.
    """
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() # Ensure that CUDA and bfloat16 are supported
    assert torch.cuda.device_count() > 0, "No CUDA devices found. Please check your setup."
    
    # Load the tokenizer and model with quantization configuration
    print(f"Loading model {model_name}...")
    print("Using bfloat16 precision for model weights.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantized_config = Mxfp4Config(dequantize = True)
    model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantized_config,
    use_cache=False,
    device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer

def wrap_model_with_LoRA(model):
    """
    Wrap the model with LoRA (Low-Rank Adaptation) for efficient training.
    
    Args:
        model: The pre-trained model to wrap.
        
    Returns:
        The wrapped model.
    """
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
    """
    Load the training configuration from a yaml file.
    Args:
        training_config_path (str): Path to the training configuration file.
        
    Returns:
        SFTConfig: The loaded training configuration.
    """
    # Ensure the training configuration file exists
    if not os.path.exists(training_config_path):
        raise FileNotFoundError(f"Training configuration file not found: {training_config_path}")
    
    # Load the training configuration from the yaml file
    print(f"Loading training configuration from {training_config_path}...")
    if not training_config_path.endswith('.yaml') and not training_config_path.endswith('.yml'):
        raise ValueError("Training configuration file must be a YAML file.")
    
    # Load and parse the YAML configuration
    with open(training_config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract training parameters from the configuration
    training_params = config_dict.get('training', {})
    model_params = config_dict.get('model', {})
    scheduler_params = config_dict.get('scheduler', {})
    logging_params = config_dict.get('logging', {})
    output_params = config_dict.get('output', {})
    
    # Create SFTConfig with the loaded parameters
    training_args = SFTConfig(
        # Training parameters
        learning_rate=float(training_params.get('learning_rate', 2e-5)),
        per_device_train_batch_size=int(training_params.get('per_device_train_batch_size', 1)),
        gradient_accumulation_steps=int(training_params.get('gradient_accumulation_steps', 1)),
        gradient_checkpointing=training_params.get('gradient_checkpointing', False),
        num_train_epochs=int(training_params.get('num_train_epochs', 3)),
        warmup_ratio=float(training_params.get('warmup_ratio', 0.03)),
        # Model parameters - using max_length from SFTConfig
        max_length=model_params.get('max_length', 512),
        
        # Scheduler parameters
        lr_scheduler_type=scheduler_params.get('lr_scheduler_type', 'linear'),
        lr_scheduler_kwargs= scheduler_params.get('lr_scheduler_kwargs', {}),
        
        # Logging parameters
        logging_steps=logging_params.get('logging_steps', 1),
        report_to=logging_params.get('report_to', "mlflow"),
        run_name=logging_params.get('run_name', "gpt-oss-20b-multilingual-reasoner")+f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        
        # Output parameters
        output_dir=output_params.get('output_dir', './results'),
        push_to_hub=output_params.get('push_to_hub', False),
        
    )
    
    return training_args

def main():
    # Load the dataset
    dataset_name = "HuggingFaceH4/Multilingual-Thinking"
    train_dataset = load_dataset_from_hf(dataset_name, 'train')
    
    # Load the model and tokenizer
    model_name = "openai/gpt-oss-20b"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Wrap the model with LoRA
    peft_model = wrap_model_with_LoRA(model)
    
    # Load the training configuration
    training_config_path = "config/training_config.yaml"
    training_args = load_training_config(training_config_path)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    # Save the trained model
    trainer.save_model(training_args.output_dir)


    
if __name__ == "__main__":
    main()
    print("Training script executed successfully.")