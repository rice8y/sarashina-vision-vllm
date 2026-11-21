def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.tokenizers import TokenizerRegistry

    from .configuration import SarashinaVisionConfig

    architecture = "Sarashina2VisionForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        AutoConfig.register(SarashinaVisionConfig.model_type, SarashinaVisionConfig)
        ModelRegistry.register_model(
            architecture, "sarashina_vision_vllm.model:Sarashina2VisionForCausalLM"
        )
        TokenizerRegistry.register(
            "sarashina", "sarashina_vision_vllm.tokenizer", "SarashinaTokenizer"
        )
