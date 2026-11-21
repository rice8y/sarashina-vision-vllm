from importlib.resources import files

_LEGACY_CONTENT = (
    files(__package__).joinpath("legacy_content.jinja").read_text(encoding="utf-8")
)


def get_chat_template(tokenizer) -> str:
    """Adapt the checkpoint template to OpenAI-format image and text messages."""
    template = tokenizer.get_chat_template()
    if "### Human:" in template:
        return _LEGACY_CONTENT + template
    return template
