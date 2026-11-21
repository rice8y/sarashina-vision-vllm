# Sarashina Vision for vLLM

**Unofficial** [vLLM](https://docs.vllm.ai/) plugin for [Sarashina2-Vision 8B](https://huggingface.co/sbintuitions/sarashina2-vision-8b), [14B](https://huggingface.co/sbintuitions/sarashina2-vision-14b), and [Sarashina2.2-Vision 3B](https://huggingface.co/sbintuitions/sarashina2.2-vision-3b).

## Install

```bash
pip install git+https://github.com/rice8y/sarashina-vision-vllm.git
```

Add `--no-deps` if a [compatible vLLM version](pyproject.toml) is already installed.

## Offline Inference

```python
from PIL import Image
from vllm import LLM, SamplingParams

from sarashina_vision_vllm.chat import get_chat_template

model = "sbintuitions/sarashina2-vision-8b"
llm = LLM(model, tokenizer_mode="sarashina")
tokenizer = llm.get_tokenizer()

image = Image.open("kaonashi.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "このキャラクターの名前は何ですか？"},
        ],
    }
]
prompt = tokenizer.apply_chat_template(
    messages,
    chat_template=get_chat_template(tokenizer),
    tokenize=False,
    add_generation_prompt=True,
)
result = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    SamplingParams(temperature=0, max_tokens=256, stop=["\n###"]),
)[0]
print(result.outputs[0].text)
```

`tokenizer_mode="sarashina"` loads the checkpoint's Unigram tokenizer directly.
The default Llama tokenizer loader in Transformers v5 can load it as BPE and produce different token IDs.

Keep `trust_remote_code` disabled. For Sarashina2.2-Vision 3B, omit `stop`.
For text-only input, omit the image part and `multi_modal_data`.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The plugin is based on the [SB Intuitions reference implementation](https://huggingface.co/sbintuitions/sarashina2-vision-8b/blob/main/modeling_sarashina2_vision.py).
