from huggingface_hub import hf_hub_download

REPO_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json"
]

# Скачать все файлы
for file in FILES:
    hf_hub_download(
        repo_id=REPO_ID,
        filename=file,
        local_dir="./models/deepseek-1.5b",
        local_dir_use_symlinks=False
    )
