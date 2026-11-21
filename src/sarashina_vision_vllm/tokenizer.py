from pathlib import Path

from transformers import PreTrainedTokenizerFast
from vllm.tokenizers.hf import get_cached_tokenizer


class SarashinaTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        download_dir: str | None = None,
        **kwargs,
    ) -> PreTrainedTokenizerFast:
        # AutoTokenizer can rebuild the tokenizer as Llama and lose Sarashina's merges.
        kwargs.setdefault("cache_dir", download_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            path_or_repo_id, *args, **kwargs
        )
        return get_cached_tokenizer(tokenizer)
