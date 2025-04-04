# Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models

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
Name the data as `rankwogpt-data.json`


### Run SFT
Create a `.yaml` file according to the base model you want to run on. `config/qwen2_5_7B_lora_rankwogpt_ddp.yaml` is an example file for `Qwen/Qwen2.5-7B-Instruct`.
```
tune run --nnodes 1 --nproc_per_node 4 lora_finetune_distributed --config config/qwen2_5_7B_lora_rankwogpt_ddp.yaml
```
`torchtune` provides [configuration examples](https://github.com/pytorch/torchtune/tree/main/recipes/configs) for more base models.  Change the **model path** and **data configurations** as in the `.yaml` config files. 


# Inference

### Installation 
```
pip install -r inference/requirements.txt
```

### Run Sliding Window
```
python sliding_window.py -device 1 --temperature 0 --seed 42 \
    -d msmarco-passage/trec-dl-2019 \
    -window 20 -step 10 \
    -model codellama/CodeLlama-13b-Instruct-hf # <-- change to out-of-box huggingface models or models reproduced by above SFT step 
```
where `device` controls how many GPU devices to run on.



# Citation
If you find the paper or code helpful, please kindly cite: 
```
@InProceedings{zhang2025rankwogpt,
author="Zhang, Crystina and Hofst{\"a}tter, Sebastian and Lewis, Patrick and Tang, Raphael and Lin, Jimmy",
editor="Hauff, Claudia and Macdonald, Craig and Jannach, Dietmar and Kazai, Gabriella and Nardini, Franco Maria and Pinelli, Fabio and Silvestri, Fabrizio and Tonellotto, Nicola",
title="Rank-Without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models",
booktitle="Advances in Information Retrieval",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="233--247",
isbn="978-3-031-88711-6"
}
```