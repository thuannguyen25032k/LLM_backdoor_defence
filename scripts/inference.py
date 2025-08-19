from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Inference script for LLM backdoor defense")
    parser.add_argument("--peft_model_id", type=str, required=True, help="ID of the PEFT model")
    parser.add_argument("--model_name", type=str, required=False, default="openai/gpt-oss-20b", help="Name of the pre-trained model")
    return parser.parse_args()

def run_inference(system_prompt, user_prompt, reasoning_language="German"):
    messages = [
        {"role": "system", "content": f"reasoning language: {reasoning_language}"},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6}
    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    return response



if __name__ == "__main__":
    args = get_args()
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).cuda()
    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, args.peft_model_id)
    model = model.merge_and_unload()  # Merge the PEFT model into the base model

    # Inference with German reasoning
    out = run_inference("reasoning language: German", "¿Cuál es el capital de Australia?", "German")
    print("\nGerman reasoning output:\n", out)


    # Inference with Chinese reasoning (not seen in finetuning)
    out_chinese = run_inference("reasoning language: Chinese", "What is the national symbol of Canada?", "Chinese")
    print("\nChinese reasoning output:\n", out_chinese)

