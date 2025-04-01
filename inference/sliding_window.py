import os
import copy
import argparse
from tqdm import tqdm
from nirtools.ir import write_runs

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from data import load_data
from evaluate import evaluate


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def get_prefix_prompt(query, num):
    return [
            {'role': 'user', 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
    ]


def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    max_length = 300 # max length per content

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()

        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response: list):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])

    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        for k in ['rank', 'score']: # do not need to rewrite docid
            if k in item['hits'][j + rank_start]:
                item['hits'][j + rank_start][k] = cut_range[j][k]

    return item 


def run_llm(messages, model: LLM, lora_request: LoRARequest = None):
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n", "")
    prompts = [prompt]
    if lora_request is not None:
        outputs = model.generate(prompts, sampling_params, use_tqdm=False, lora_request=lora_request)
    else:
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

    return generated_text


def permutation_pipeline(model: LLM, lora_request: LoRARequest, item=None, rank_start=0, rank_end=100):
    # TODO: instruction might need to be changed
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
    permutation = run_llm(messages, model, lora_request) # text
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(model: LLM, lora_request: LoRARequest, item=None, rank_start=0, rank_end=100, window_size=20, step=10): 
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(model, lora_request, item, start_pos, end_pos)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", "-lora", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str, default="msmarco-passage/trec-dl-2019")
    parser.add_argument("--window_size", "-window", type=int, default=20)
    parser.add_argument("--step", "-step", type=int, default=10)
    parser.add_argument("--output_dir", "-o", type=str, default="rerank-results/") 

    # post-processing of arguments
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    lora_path = args.lora_path
    dataset = args.dataset

    config_path = f"window-{args.window_size}-step-{args.step}"
    model_path = os.path.basename(model_name)
    if lora_path:
        lora_base_path = ".".join(lora_path.strip("/").split("/")[-2:])
        model_path += f".LORA-{lora_base_path}"

    dataset_path = os.path.basename(dataset)
    output_dir = os.path.join(
        output_dir,
        model_path, config_path, dataset_path,
    )
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    return args


def get_lora_request(model_name, lora_path):
    return LoRARequest(
        lora_name="lora",
        lora_int_id=1,
        lora_path=lora_path,
        base_model_name=model_name,
    )


def main(args):
    dataset, model_name, lora_path = args.dataset, args.model_name, args.lora_path
    window_size, step = args.window_size, args.step
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, "rank-wo-gpt.trec")
    if os.path.exists(output_file):
        print(f"Reranked runs already exist in {output_file}")
        evaluate(output_file, dataset)
        return
    
    print(f"Saving reranked runs to {output_file}...")

    if not lora_path:
        model = LLM(model=model_name, tokenizer=model_name)
        lora_request = None
    else:
        model = LLM(model=model_name, tokenizer=model_name, enable_lora=True)
        lora_request = get_lora_request(model_name, lora_path)

    runs, qid2query, docid2doc = load_data(dataset=dataset)

    reranked_runs = {}
    for qid in tqdm(runs):
        query = qid2query[qid]
        docids = runs[qid]
        num_docs = len(docids)

        item = {
            'query': query,
            'hits': [
                {
                    'content': docid2doc[docid],
                    'docid': docid,
                    'rank': rank,
                } for rank, docid in enumerate(runs[qid])
            ]
        }
        item = sliding_windows(model, lora_request, item, rank_start=0, rank_end=num_docs, window_size=window_size, step=step)
        docid2rank = {hit['docid']: hit['rank'] for hit in item['hits']}
        reranked_runs[qid] = {docid: -rank for docid, rank in docid2rank.items()} # higher the rank, lower the score

    write_runs(reranked_runs, output_file)
    print(f"Reranked runs written to {output_file}")
    evaluate(output_file, dataset)

    return reranked_runs



if __name__ == "__main__":
    args = get_args()
    main(args)
