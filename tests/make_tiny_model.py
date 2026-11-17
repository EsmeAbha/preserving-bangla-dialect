"""Builds a tiny randomly initialised Qwen2-style chat model offline, to smoke-test generate.py and
train_qlora.py on CPU (python train_qlora.py --model <dir> --no_4bit --max_steps 4 ...).
Its outputs are meaningless."""
import argparse
import json

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from common import INSTRUCTIONS, load_jsonl

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = [r["src"] + " " + r["tgt"] for r in load_jsonl(f"{args.data_dir}/train.jsonl")] + INSTRUCTIONS
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=600, special_tokens=[
        "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHATML
    tok.save_pretrained(args.out)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(args.out)
    print(json.dumps({"vocab": len(tok), "dir": args.out}))


if __name__ == "__main__":
    main()
