"""Loading a 4-bit causal LM and batched greedy generation with the model's chat template."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import build_messages, clean_output

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",  # backup if GPU time runs out
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",  # gated: accept the licence on HF and set HF_TOKEN
}


def compute_dtype():
    if not torch.cuda.is_available():
        return torch.bfloat16  # CPU: bf16 is fast with AMX/AVX-512 and fp16 is not
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id, adapter=None, four_bit=True):
    dtype = compute_dtype()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if four_bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, dtype=dtype,
                                                 device_map={"": 0} if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def stop_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i is not None and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def prompt_text(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, rows, batch_size=16, max_new_tokens=128, instruction_id=0, with_dialect=True,
             shot_sampler=None, progress=True):
    """Greedy decoding; returns rows with `raw` (model text) and `hyp` (cleaned output)."""
    tok.padding_side = "left"
    eos = stop_ids(tok)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["src"]))  # similar lengths per batch
    raw = [None] * len(rows)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        texts = [prompt_text(tok, build_messages(rows[i], instruction_id, with_dialect,
                                                 shot_sampler(rows[i]) if shot_sampler else ()))
                 for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        for i, seq in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            raw[i] = tok.decode(seq, skip_special_tokens=True)
        if progress and (b // batch_size) % 20 == 0:
            print(f"  generated {min(b + batch_size, len(rows))}/{len(rows)}", flush=True)
    return [{**r, "raw": raw[i], "hyp": clean_output(raw[i])} for i, r in enumerate(rows)]
