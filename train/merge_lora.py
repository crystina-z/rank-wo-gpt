#!/usr/bin/env python
"""
merge_and_push_peft_model.py

Usage:
  python merge_and_push_peft_model.py \
      --base_model_name base/model \
      --peft_model_name your-username/peft-model \
      --auth_token YOUR_HF_TOKEN
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(base_model_name, peft_model_name, auth_token=None):
    # 1. Load the base model
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # adjust dtype as needed
        device_map="auto",  # optional if using GPU / accelerate
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2. Load the PEFT (adapter) weights
    print(f"Loading PEFT adapter from: {peft_model_name}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        peft_model_name,
        torch_dtype=torch.float16,  # match base model dtype
    )

    # 3. Merge LoRA/PEFT adapter weights into the base model
    #    (for LoRA, `merge_and_unload` merges the LoRA weights and returns a regular model)
    print("Merging PEFT adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # Optionally, re-instantiate the tokenizer from the PEFT model folder if there were changes
    # In most cases, the tokenizer is identical to the base, so it's not strictly required.
    # tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

    # Define a new model name with suffix '-merged'
    merged_model_name = f"{peft_model_name}-merged"
    print(f"Pushing merged model to: {merged_model_name}")


    # # 4. Push to Hugging Face Hub
    # merged_model.push_to_hub(merged_model_name, use_auth_token=auth_token)
    # # Also push the tokenizer
    # tokenizer.push_to_hub(merged_model_name, use_auth_token=auth_token)
    # print("Successfully pushed merged model to the Hub!")

    merged_model.save_pretrained(merged_model_name)
    tokenizer.save_pretrained(merged_model_name)
    print(f"Successfully saved merged model to {merged_model_name}!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge PEFT adapter with a base model and push to HF Hub with '-merged' suffix."
    )
    parser.add_argument(
        "--base_model_name", "-base", type=str, required=True, help="Path or ID of the base model on Hugging Face."
    )
    parser.add_argument(
        "--peft_model_name", "-peft", type=str, required=True, help="Path or ID of the PEFT fine-tuned model on Hugging Face."
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        required=False,
        help="Your Hugging Face API token. If omitted, must be logged in via CLI.",
    )
    args = parser.parse_args()

    main(base_model_name=args.base_model_name, peft_model_name=args.peft_model_name, auth_token=args.auth_token)