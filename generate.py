import torch

def greedy_decoding(model, tokenizer, prompt="", max_new_tokens=1000, device="cpu"):
  tokenized_prompt = tokenizer(prompt, return_tensors="pt")
  generated_ids = tokenized_prompt.input_ids.to(device)
  model = model.to(device)

  max_new_tokens += tokenized_prompt.input_ids.shape[1]
  next_token_id = torch.tensor(0)
  while next_token_id.item() != tokenizer.eos_token_id and generated_ids.shape[1] < max_new_tokens:
    with torch.no_grad():
      outputs = model(input_ids=generated_ids)
      logits = outputs.logits

    next_token_logits = logits[:, -1 :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

  return generated_ids.view(-1)


def sampling(model, tokenizer, prompt="", max_new_tokens=1000, device="cpu"):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    generated_ids = tokenized_prompt.input_ids.to(device)
    model = model.to(device)

    max_new_tokens += tokenized_prompt.input_ids.shape[1]
    next_token_id = torch.tensor(0)
    while next_token_id.item() != tokenizer.eos_token_id and generated_ids.shape[1] < max_new_tokens:
      with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits


      probs = torch.softmax(logits, dim=-1)
      next_token_id = torch.multinomial(probs[:, -1, :], num_samples=1)

      generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    return generated_ids.view(-1)


def temperature_sampling(model, tokenizer, prompt="", max_new_tokens=1000, temperature=1.0, device="cpu"):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    generated_ids = tokenized_prompt.input_ids.to(device)
    model = model.to(device)

    max_new_tokens += tokenized_prompt.input_ids.shape[1]
    next_token_id = torch.tensor(0)
    while next_token_id.item() != tokenizer.eos_token_id and generated_ids.shape[1] < max_new_tokens:
      with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits


      probs = torch.softmax(logits/temperature, dim=-1)
      next_token_id = torch.multinomial(probs[:, -1, :], num_samples=1)

      generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    return generated_ids.view(-1)


def nucleus_search(model, tokenizer, prompt="", max_new_tokens=1000, temperature=1.0, top_p=1.0, device='cpu'):
    assert top_p > 0.0
    assert top_p <= 1.0

    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    generated_ids = tokenized_prompt.input_ids.to(device)
    model = model.to(device)

    max_new_tokens += tokenized_prompt.input_ids.shape[1]
    next_token_id = torch.tensor(0)
    while next_token_id.item() != tokenizer.eos_token_id and generated_ids.shape[1] < max_new_tokens:
      with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits


      sorted_logits, sorted_indices = logits.sort(descending=True)
      probs = torch.softmax(sorted_logits/temperature, dim=-1)

      cummulative_probs = torch.cumsum(probs, dim=-1)

      if cummulative_probs[0][-1][0] > top_p:
        top_p_index = 1
      else:
        top_p_index = torch.nonzero(cummulative_probs[0][-1] > top_p, as_tuple=False)[0].item()


      indices_to_remove = sorted_indices[0, :, top_p_index:]
      sorted_logits[0][-1][indices_to_remove[-1]] = float('-inf')

      real_probs = torch.softmax(sorted_logits/temperature, dim=-1)
      next_token_id = torch.multinomial(real_probs[:, -1, :], num_samples=1)

      generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    return generated_ids.view(-1)


def beam_search(model, tokenizer, prompt="", max_new_tokens=1000, temperature=1.0, num_beams=3, length_penalty=1, device='cpu'):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    generated_ids = tokenized_prompt.input_ids.to(device)
    model = model.to(device)

    max_new_tokens += tokenized_prompt.input_ids.shape[1]

    candidates = []
    finished_candidates = []

    while len(finished_candidates) < num_beams:
      if not candidates and not finished_candidates:
        with torch.no_grad():
          outputs = model(input_ids=generated_ids)
          logits = outputs.logits

        k_candidates = torch.topk(logits[0][-1], num_beams)

        candidates_tokens = k_candidates.indices.unsqueeze(-1).detach().cpu()
        candidates_probs = k_candidates.values.detach().cpu().tolist()

        candidates = [[token, score] for token, score in zip(candidates_tokens, candidates_probs) if token != tokenizer.eos_token_id]
        finished_candidates = [[token, score] for token, score in zip(candidates_tokens, candidates_probs) if token == tokenizer.eos_token_id]

      candidates2 = []
      for i, candidate in enumerate(candidates):
        token = candidate[0]
        score = candidate[1]

        tokenized_input = torch.cat([generated_ids, token.unsqueeze(0).to(device)], -1).to(device)

        with torch.no_grad():
          outputs = model(input_ids=tokenized_input)
          logits = outputs.logits

        k_candidates = torch.topk(logits[0][-1], num_beams)
        candidates_tokens_temp = k_candidates.indices.unsqueeze(-1).detach().cpu()
        candidates_probs_temp = k_candidates.values.detach().cpu()

        candidates_temp = [[torch.cat([token, token1], dim=-1), score + score1.item()] for token1, score1 in zip(candidates_tokens_temp, candidates_probs_temp)]

        for cndt in candidates_temp:
          if cndt[0][-1] == tokenizer.eos_token_id or len(cndt[0]) > max_new_tokens:
            finished_candidates.append(cndt)
          else:
            candidates2.append(cndt)

      candidates = sorted(candidates2, key=lambda candidate: candidate[1] / len(candidate[0]) ** length_penalty, reverse=True)[:num_beams+1]

    return sorted(finished_candidates, key=lambda candidate: candidate[1] / len(candidate[0]) ** length_penalty, reverse=True)