# Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models

<div align="center">
  <img src="rankwogpt.png" width="400">
</div>

Disclaimer: This is a *reimplementation* of [Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models](https://link.springer.com/chapter/10.1007/978-3-031-88711-6_15).

This repository contains code and scripts to run SFT and inference of listwise reranking.


# SFT 

### Installation 
```
pip install -r train/requirements.txt
```

### Download Backbone
```
hgf_model_name=Qwen/Qwen2.5-7B-Instruct
local_model_dir=model
tune download $hgf_model_name --output-dir model/$(basename $hgf_model_name)
```

### Download Training Data
Name the data as `rankwogpt-data.json`. A sample of training data format is given in `rankwogpt-data.sample.json`. 
Data preparing script will be reason soon.


### Run SFT
Create a `.yaml` file according to the base model you want to run on. `config/qwen2_5_7B_lora_rankwogpt_ddp.yaml` is an example file for `Qwen/Qwen2.5-7B-Instruct`.
```
tune run --nnodes 1 --nproc_per_node 4 lora_finetune_distributed --config train/config/qwen2_5_7B_lora_rankwogpt_ddp.yaml
```
`torchtune` provides [configuration examples](https://github.com/pytorch/torchtune/tree/main/recipes/configs) for more base models.  Change the **model path** and **data configurations** as in the `.yaml` config files. 


# Inference

### Installation 
```
pip install -r inference/requirements.txt
```

### Run Sliding Window
```
model_path=rank-wo-gpt/qwen2_5_7B_lora.cohere-english-v2

python inference/sliding_window.py -device 1 \
    --temperature 0 --seed 42 \
    --dataset msmarco-passage/trec-dl-2019 \
    -window 20 -step 10 \
    -model ${model_path} # <-- change to out-of-box huggingface models or models reproduced by above SFT step 
    -tokenizer Qwen/Qwen2.5-7B-Instruct \ # <-- change to the tokenizer of corresponding base model
    -shuffle False # <-- switch on to test the consistency regarding shuffled document candidates
```
where `device` controls how many GPU devices to run on,
and `model_path` is the path to the merged model weights.
We removed the support of loading LORA weights due to efficiency reasons.
If you only have adapter weights, [`train/merge_lora.py`](train/merge_lora.py) provides the script to merge LORA weights into the base model.

Below is a list of checkpoints released on HuggingFace and their expected nDCG@10 scores (with `--temperature 0`)

| HuggingFace checkpiont | TREC-DL-2019 | TREC-DL-2020 |
|--|--|--|
|[rank-wo-gpt/qwen2_5_3B_full.cohere-english-v2](https://huggingface.co/rank-wo-gpt/qwen2_5_3B_full.cohere-english-v2)| 0.656 | 0.573 |
|[rank-wo-gpt/qwen2_5_7B_lora.cohere-english-v2](https://huggingface.co/rank-wo-gpt/qwen2_5_7B_lora.cohere-english-v2)| 0.732 | 0.696 |
|[rank-wo-gpt/llama3_1_8B_lora.cohere-english-v2](https://huggingface.co/rank-wo-gpt/llama3_1_8B_lora.cohere-english-v2)| 0.739	| 0.651 | 



### Consistency Regarding Shuffled Inputs 
Although we observed that newer models (e.g., Qwen) are rather robust againt the shuffled inputs,
we provied script to run [Permutation Self-Consistency](https://github.com/castorini/perm-sc) on ranking task.
To run perm-sc, first [install the package following their description](https://github.com/castorini/perm-sc?tab=readme-ov-file#installation).
Then:
```
python inference/psc_inference.py -device 1 \
    --num_seeds 20 \ # <-- number of permutations to aggregate
    --temperature 0.25 \
    --dataset msmarco-passage/trec-dl-2019 \
    -window 20 -step 10 \
    -model ${model_path} # <-- change to out-of-box huggingface models or models reproduced by above SFT step 
    -tokenizer Qwen/Qwen2.5-7B-Instruct \ # <-- change to the tokenizer of corresponding base model
    -shuffle False # <-- switch on to test the consistency regarding shuffled document candidates
```


# Citation
If you find the paper or code helpful, please kindly cite: 
```
@InProceedings{zhang2025rankwogpt,
    author="Zhang, Crystina and Hofst{\"a}tter, Sebastian and Lewis, Patrick and Tang, Raphael and Lin, Jimmy",
    editor="Hauff, Claudia and Macdonald, Craig and Jannach, Dietmar and Kazai, Gabriella and Nardini, Franco Maria and Pinelli, Fabio and Silvestri, Fabrizio and Tonellotto, Nicola",
    title="Rank-Without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models",
    booktitle="Advances in Information Retrieval",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="233--247",
    isbn="978-3-031-88711-6"
}
```