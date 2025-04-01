import copy
import argparse
from tqdm import tqdm
from nirtools.ir import write_runs

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from data import load_data
from pprint import pprint


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
        # messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
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


from vllm.lora.request import LoRARequest

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/mnt/users/x978zhan/paper-backup/rank-wo-gpt/after-acceptance-ECIR/rank-wo-gpt/inference/checkpoints/epoch_2"
LORA = LoRARequest(
    lora_name="lrl-lora",
    lora_int_id=1,
    lora_path=LORA_PATH,
    base_model_name=base_model_name,
)

def run_llm(messages, model: LLM):
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n", "")
    prompts = [prompt]
    outputs = model.generate(prompts, sampling_params, use_tqdm=False, lora_request=LORA)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return generated_text


def permutation_pipeline(model: LLM, item=None, rank_start=0, rank_end=100):
    # TODO: instruction might need to be changed
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
    permutation = run_llm(messages, model) # text
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(model: LLM, item=None, rank_start=0, rank_end=100, window_size=20, step=10): 
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(model, item, start_pos, end_pos)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019")
    return parser.parse_args()


def main(model_name, dataset):
    model = LLM(model=model_name, tokenizer=model_name, enable_lora=True)
    model_name += "-epoch-2-lora"

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
        item = sliding_windows(model, item, rank_start=0, rank_end=num_docs, window_size=20, step=10)
        docid2rank = {hit['docid']: hit['rank'] for hit in item['hits']}
        reranked_runs[qid] = {docid: -rank for docid, rank in docid2rank.items()} # higher the rank, lower the score

    runfile = f"data/{model_name}/{dataset}/reranked_runs.update-prompt.trec"
    write_runs(reranked_runs, runfile)
    print(f"Reranked runs written to {runfile}")

    return reranked_runs



if __name__ == "__main__":
    dataset = "msmarco-passage/trec-dl-2019"
    dataset = "msmarco-passage/trec-dl-2020"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    main(model_name, dataset)