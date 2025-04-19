import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import greedy_decoding, sampling, temperature_sampling, nucleus_search, beam_search



if __name__ == "__main__":
    device = "cpu"

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    encoding = tokenizer(input_text_hedgehog, return_tensors="pt")
    json_encodding = tokenizer(input_text_json, return_tensors="pt")

    # print("started")

    greed_sonic = greedy_decoding(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000, device=device)
    greed_sonic = tokenizer.decode(greed_sonic[encoding.input_ids.shape[1]:])
    greed_json = greedy_decoding(model, tokenizer, prompt=input_text_json, max_new_tokens=1000, device=device)
    greed_json = tokenizer.decode(greed_json[json_encodding.input_ids.shape[1]:])

    sampled_sonic = sampling(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000, device=device)
    sampled_sonic = tokenizer.decode(sampled_sonic[encoding.input_ids.shape[1]:])
    sampled_json = sampling(model, tokenizer, prompt=input_text_json, max_new_tokens=1000, device=device)
    sampled_json = tokenizer.decode(sampled_json[json_encodding.input_ids.shape[1]:])



    temperatures = [0.001, 0.1, 0.5, 1.0, 10.0]
    sampled_sonic_texts = []
    sampled_json_texts = []

    for temperature in temperatures:
        sonic_text = temperature_sampling(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000,
                                          temperature=temperature, device=device)
        sonic_text = tokenizer.decode(sonic_text[encoding.input_ids.shape[1]:])
        sampled_sonic_texts.append(sonic_text)

        json_text = temperature_sampling(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000,
                                         temperature=temperature, device=device)
        json_text = tokenizer.decode(json_text[encoding.input_ids.shape[1]:])
        sampled_json_texts.append(json_text)

    params = [(1, 0.9), (1, 0.15), (0.5, 0.9), (0.5, 0.15)]
    nucleus_sonic_texts = []
    nucleus_json_texts = []

    for (temperature, top_p) in params:
        # print(temperature, top_p)
        sonic_text = nucleus_search(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000,
                                    temperature=temperature, top_p=top_p, device=device)
        sonic_text = tokenizer.decode(sonic_text[encoding.input_ids.shape[1]:])
        nucleus_sonic_texts.append(sonic_text)

        json_text = nucleus_search(model, tokenizer, prompt=input_text_json, max_new_tokens=1000,
                                   temperature=temperature, top_p=top_p, device=device)
        json_text = tokenizer.decode(json_text[json_encodding.input_ids.shape[1]:])
        nucleus_json_texts.append(json_text)

    params = [(1, 1.0), (4, 1.0), (4, 0.5), (4, 2.0), (8, 1.0)]

    beam_sonic_texts = []
    beam_json_texts = []

    for (beams, lp) in params:
        # print(beams, lp)
        sonic_text = beam_search(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000, num_beams=beams,
                                 length_penalty=lp, device=device)
        sonic_text = tokenizer.decode(sonic_text[0][0])
        beam_sonic_texts.append(sonic_text)

        json_text = beam_search(model, tokenizer, prompt=input_text_json, max_new_tokens=1000, num_beams=beams,
                                length_penalty=lp, device=device)
        json_text = tokenizer.decode(json_text[0][0])
        beam_json_texts.append(json_text)
        
    
    # print("greed")
    # print(greed_sonic, greed_json)
    # print("sample")
    # print(sampled_sonic, sampled_json)
    # print("temperature")
    # print(sampled_sonic_texts, sampled_json_texts)
    # print("nucleus")
    # print(nucleus_sonic_texts, nucleus_json_texts)
    # print("beam")
    # print(beam_sonic_texts, beam_json_texts)