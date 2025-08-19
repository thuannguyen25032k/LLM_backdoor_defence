# LLM Backdoor Defense

A comprehensive framework for fine-tuning GPT-OSS with built-in backdoor defense mechanisms. This project focuses on training robust language models for implementing security measures against potential backdoor attacks.

## 🎯 Overview

This project provides tools and scripts for:
- Fine-tuning GPT-style models with LoRA (Low-Rank Adaptation)
- Implementing backdoor defense mechanisms with training dataset against backdoor attacks
- Monitoring training progress with MLflow integration
- Configurable training pipelines through YAML configurations

## 📁 Project Structure

```
LLM_backdoor_defence/
├── README.md                    # This file
├── LICENSE                      # Project license
├── requirements.txt             # Python dependencies
├── fine-tune-transfomers.ipynb  # Jupyter notebook for interactive training
├── config/                      # Configuration files
│   ├── model_config.json        # Model configuration
│   ├── tokenizer_config.json    # Tokenizer configuration
│   └── training_config.yaml     # Training hyperparameters
├── data/                        # Data directory
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets
│   └── prompts/                 # Training prompts
├── scripts/                     # Main training scripts
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Model evaluation
│   ├── inference.py             # Model inference
│   └── preprocess.py            # Data preprocessing
├── experiments/                 # Experiment results
├── results/                     # Training outputs
└── mlruns/                     # MLflow tracking data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access token

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/thuannguyen25032k/LLM_backdoor_defence.git
cd LLM_backdoor_defence
```

2. **Set up Python environment:**
```bash
# Using conda (recommended)
conda create -n llm-defense python=3.10.18
conda activate llm-defense
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

# Install additional dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

4. **Set up Hugging Face authentication:**
```bash
# Login command
hf auth login
```

## ⚙️ Configuration

### Training Configuration

Edit `config/training_config.yaml` to customize your training:

```yaml
model:
  name: "gpt-oss"
  max_length: 2048

training:
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  num_train_epochs: 1
  warmup_ratio: 0.03

scheduler:
  lr_scheduler_type: "cosine_with_min_lr"
  lr_scheduler_kwargs:
    min_lr_rate: 0.1

logging:
  logging_steps: 1
  report_to: "mlflow"

output:
  output_dir: "results/gpt-oss-20b-multilingual-reasoner" # need to be modified
  push_to_hub: false

experiment:
  name: "gpt-oss-finetuning"
  tracking_uri: "./mlruns"
```

## 📊 Monitoring Training Progress 

### MLflow Integration

1. **Start MLflow UI** (in a separate terminal):
```bash
cd LLM_backdoor_defence
# Firstly, Run mlflow UI for tracking training process
mlflow server --host 127.0.0.1 --port 8080

# Logging to the Tracking Server
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
```

2. **Access the UI:**
   - Open browser to `http://localhost:8080`
   - View experiment metrics, parameters, and artifacts
   - Compare different training runs

### Key Metrics Tracked:
- Training loss
- Learning rate schedule
- GPU memory usage
- Training speed (tokens/second)
- Model parameters and gradients

## 🏃‍♂️ Running the Training

### Method 1: Using Python Script

```bash

# Run training with default configuration
python scripts/train.py

# Run inference with default configuration
python scripts/inference.py -- peft_model_id results/gpt-oss-20b-multilingual-reasoner
```

### Method 2: Using Jupyter Notebook

```bash
# Start Jupyter server
jupyter notebook

# Open fine-tune-transfomers.ipynb
# Follow the notebook cells step by step
```

## 🔧 Advanced Usage

### Custom Dataset Preparation

1. **Prepare your dataset:**
```python
from datasets import Dataset
import pandas as pd

# Load your data
data = pd.read_csv('your_dataset.csv')
dataset = Dataset.from_pandas(data)

# Format for training (text completion format)
def format_dataset(example):
    return {"text": f"Input: {example['input']}\nOutput: {example['output']}"}

formatted_dataset = dataset.map(format_dataset)
```

2. **Update the dataset loading in `scripts/train.py`:**
```python
# Replace the dataset loading section
dataset = load_dataset("path/to/your/dataset", split="train")
```

### LoRA Configuration

Customize LoRA parameters in `scripts/train.py`:
```python
peft_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,         # Alpha parameter
    target_modules="all-linear",  # Target modules
    lora_dropout=0.1,      # Dropout
    bias="none",           # Bias
    task_type="CAUSAL_LM"  # Task type
)
```

## 🛡️ Backdoor Defense Features

### Defense Mechanisms Implemented:

1. **Data Sanitization:**
   - Automatic detection of suspicious patterns
   - Data cleaning and validation

2. **Training Monitoring:**
   - Gradient analysis for anomaly detection
   - Loss pattern analysis

3. **Model Validation:**
   - Post-training model behavior analysis
   - Trigger detection tests

### Running Defense Tests:

```bash
# Run backdoor detection
python scripts/evaluate.py --mode backdoor_detection

# Run model validation
python scripts/evaluate.py --mode validation
```

## 📈 Evaluation and Inference

### Model Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model_path ./results/checkpoint-1000

# Run specific evaluation tasks
python scripts/evaluate.py --tasks perplexity,generation_quality
```

### Inference

```bash
# Interactive inference
python scripts/inference.py --model_path ./results/final_model

# Batch inference
python scripts/inference.py --input_file ./data/test_prompts.txt --output_file ./results/predictions.txt
```

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size in config/training_config.yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 8
   ```

2. **Hugging Face Authentication Error:**
   ```bash
   # Re-authenticate
   huggingface-cli logout
   huggingface-cli login
   ```

3. **Model Not Found:**
   - Check if the model name is correct
   - Ensure you have access to private models
   - Try with a public model first

4. **Memory Issues:**
   ```bash
   # Enable gradient checkpointing
   gradient_checkpointing: true
   
   # Use smaller model or sequence length
   max_length: 512
   ```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- The research community for backdoor defense methodologies
- MLflow for experiment tracking capabilities

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Contact the maintainers
- Check the documentation and troubleshooting section

---

**Note:** This project is for research and educational purposes. Please ensure responsible use of AI technologies.