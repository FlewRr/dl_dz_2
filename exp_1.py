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

    greed_sonic = greedy_decoding(model, tokenizer, prompt=input_text_hedgehog, max_new_tokens=1000, device=device)
    greed_sonic = tokenizer.decode(greed_sonic[encoding.input_ids.shape[1]:])
    greed_json = greedy_decoding(model, tokenizer, prompt=input_text_json, max_new_tokens=1000, device=device)
    greed_json = tokenizer.decode(greed_json[json_encodding.input_ids.shape[1]:])


    print(greed_sonic)
    print(greed_json)