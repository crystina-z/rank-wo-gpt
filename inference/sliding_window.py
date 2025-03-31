import copy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

query = "what does time and a half mean"
# PROMPT_TEMPLATE = """I will provide you with 20 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: what does time and a half mean.

PROMPT_TEMPLATE = """
[1] What does a drug's half-life mean? Half-life is the period of time it takes for a substance undergoing decay to decrease by half. A drug's shelf-life is determined by finding out how long it takes a medication or drug to be eliminated from blood plasma by one half of its strength.
[2] solved What does timing mean on RAM? solved What does RAM Speed(OC) actually mean for Motherboard specification? solved What does the RAM speed mean exactly? solved What does the blue ram slot mean in 4 slot ram? solved What does CL15 mean on ram? solved New MB-Ram, Dont know what lights mean; solved Does adding more RAM mean faster loading times in games? solved What does it mean when Dynamic RAM needs to be refreshed and why does it need to be refreshed? solved What does oc mean for ram? solved What does it mean when the power light blinks 5 times? solved my hp laptop blinks the caps lock LED up to seven times and tirps off immediately, what could that possibly mean? solved What does B1 mean in Kingston Ram? solved What Does The Numbers In RAM Mean? solved Which RAM is better
[3] Half Duplex (HDX) Definition - What does Half Duplex (HDX) mean? Half duplex (HDC) is a type of system that enables bidirectional data or voice communication between two nodes, where both end nodes send or receive data one node at a time.
[4] What is the meaning of HASTE MRI abbreviation? The meaning of HASTE MRI abbreviation is half-Fourier acquisition single-shot turbo spin-echo magnetic resonance imaging. Q: A: What is HASTE MRI abbreviation? One of the definitions of HASTE MRI is half-Fourier acquisition single-shot turbo spin-echo magnetic resonance imaging. Q: A: What does HASTE MRI mean? HASTE MRI as abbreviation means half-Fourier acquisition single-shot turbo spin-echo magnetic resonance imaging. Q: A: What is shorthand of half-Fourier acquisition single-shot turbo spin-echo magnetic resonance imaging? The most common shorthand of half-Fourier acquisition single-shot turbo spin-echo magnetic resonance imaging is HASTE MRI.
[5] Updated March 07, 2016. A medication's biological or terminal half-life is how long it takes for half of the dose to be eliminated from the bloodstream. In medical terms, the half-life of a drug is the time it takes for the plasma concentration of a drug to reach half of its original concentration. So what does that mean for dosage and medication use? Your diagnosis may be just one component of how your treatment is handled. Your body has a say as well.
[6] solved What does the toms in toms hardware mean? solved What does this issue mean? solved What does the blue ram slot mean in 4 slot ram? solved What does dx12 mean and do? PassMark Performance Test: What Does It Really Tell? solved What does it mean to put bios onto root directory of a usb? solved What does GTX 1080 mean? solved GTX1080TI PC crashing. What does it mean? solved What does it mean when the power light blinks 5 times?
[7] What is the difference between Half-life and Half-life Source? I noticed that valve puts source at the end of a game title and I want to know what does it mean. I am planing on getting Half-life original and don't know what to get.
[8] What is honesty? What does it really mean? Is a half-truth really a lie or just a half-truth? It is safe to say the definition of honesty can mean a lot of different things for a lot of different people. Wikipedia defines honesty as the human quality of communicating and acting truthfully and with fairness. Merriam
[9] THE DREADED Doomsday Clock has moved to closer to midnight today. But what is the clock and why does it symbolise the threat of global obliteration? Scientists are have moved the Doomsday Clock's hands to two and half minutes to midnight - the time representing the end of humanity. After the clock moved 30 seconds closer to midnight, here is look at what the time means and how the Doomsday clock started ticking. Related articles
[10] Report Copyright Violation. SPIRITUAL PEOPLE, what does this mean??? When someone prays to God asking for help, etc...suddenly that person lets out a big yawn. It happens almost all the time when praying or talking to God.... what does it mean?spiritual draining thats what thats called your praying with your spirit. Re: SPIRITUAL PEOPLE, what does this mean???hen someone prays to God asking for help, etc...suddenly that person lets out a big yawn. It happens almost all the time when praying or talking to God.... what does it mean? Be offended. It's my gift to you. For a few seconds, it means you're awake... Re: SPIRITUAL PEOPLE, what does this mean???
[11] Report Abusive Post. Report Copyright Violation. SPIRITUAL PEOPLE, what does this mean??? When someone prays to God asking for help, etc...suddenly that person lets out a big yawn. It happens almost all the time when praying or talking to God.... what does it mean? Re: SPIRITUAL PEOPLE, what does this mean??? did it occur to you to look up what ...hen someone prays to God asking for help, etc...suddenly that person lets out a big yawn. It happens almost all the time when praying or talking to God.... what does it mean? Be offended. It's my gift to you. For a few seconds, it means you're awake... Re: SPIRITUAL PEOPLE, what does this mean???
[12] Know what coffee is what. What a (half-drunk) caffe caffÃ¨ looks like In. Italy, obviously a latte in An american Or British starbucksâisn t the same as a latte In. (Italy since the word Is italian and does âmean,âisn t the same as a latte In. (Italy since the word Is italian and does âmean,â¦tally cruel and reallynot suited to what the actual crime was. == Over time the phrase cruel and unusual punishment has beeninterpreted many ways. In general the idea is that any punishmentimposed in a barbaric, excessive and/or bizarre manner wouldconstitute cruel and unusual punishment.
[16] Definition - What does Naga mean? Naga is a Sanskrit word meaning âsnake,âserpentâcobra.âº. 12-16 hours: Caffeine has a half-life of about 6 hours, meaning half of the caffeine will be left in your system 6 hours after ingestion, and half of whats left 6 hours later so on and so forth.. Get help from a doctor now âº. 12-16 hours: Caffeine has a half-life of about 6 hours, meaning half of the caffeine will be left in your system 6 hours after ingestion, and half of whats left 6 hours later so on and so forth.
[19] How Long Does Valium Stay in the Body? One dose of Valium has a half-life of 200 hours, which means that the medication decreases by half in the body in 8-9 days. Most benzodiazepines have a half-life of a few hours, up to one day; although this means that withdrawal takes less time, it also means that cravings for the drug begin sooner.
[20] i wanna know what does it mean when people say they have a half brother or a half sister. Add your answer. Source.
"""
# Search Query: what does time and a half mean.
# Rank the 20 passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."""


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name=''):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    # import pdb; pdb.set_trace()

    max_length = 300

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
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
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
        for keys in ['rank', 'score', 'docid']:
            if keys in item['hits'][j + rank_start]:
                item['hits'][j + rank_start][keys] = cut_range[j][keys]
        # if 'rank' in item['hits'][j + rank_start]:
        #     item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        # if 'score' in item['hits'][j + rank_start]:
        #     item['hits'][j + rank_start]['score'] = cut_range[j]['score']
        # if 'docid' in item['hits'][j + rank_start]:
        #     item['hits'][j + rank_start]['docid'] = cut_range[j]['docid']
    return item 



def run_llm(messages, model_name=''):
    # model_name = ""
    model = LLM(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256)
    # messages = [
    #     {"role": "user", "content": PROMPT_TEMPLATE}
    # ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n", "")
    prompts = [prompt]
    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # import pdb; pdb.set_trace()

    return generated_text


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name=''):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name)  # chan
    # import pdb; pdb.set_trace()
    permutation = run_llm(messages, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name=''): 
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        # import pdb; pdb.set_trace()
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


if __name__ == "__main__":
    passages = PROMPT_TEMPLATE.split('\n')
    passages = [p.strip() for p in passages if p.strip()]
    passages = [''.join(p.split(']')[1:]).strip() for p in passages]

    item = {
        'query': query,
        'hits': [
            {'content': p} for p in passages
        ]
    }

    model_name = "/u1/x978zhan/src/rank-wo-gpt/checkpoints/Qwen2_5-7B-Instruct"
    item = sliding_windows(item, rank_start=0, rank_end=20, window_size=20, step=10, model_name=model_name)
    print("\n\n\nitem", item)
