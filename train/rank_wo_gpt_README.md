### Install torchtune

```
pip install torchtune
```

### Download Backbone
```
tune download Qwen/Qwen2.5-7B-Instruct \
  --output-dir Qwen2.5-7B-Instruct
```

### Download Training Data
Name the data as `rankwogpt-data.json`

### Run SFT
```
tune run --nnodes 1 --nproc_per_node 4 lora_finetune_distributed --config qwen2_5_7B_lora_rankwogpt_ddp.yaml
```