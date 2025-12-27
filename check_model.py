from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    print(f"Architectures: {config.architectures}")
    print(f"Model Type: {config.model_type}")
except Exception as e:
    print(e)