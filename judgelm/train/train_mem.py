# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Fix: add root path to use absolute import
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))

# Need to call this before importing transformers.
from judgelm.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from judgelm.train.train import train

if __name__ == "__main__":
    train()
