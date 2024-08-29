## https://huggingface.co/tokyotech-llm/Swallow-7b-instruct-hf
print('########################################')
print('######## text-generation by Swallow')

from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tokyotech-llm/Swallow-7b-instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")

PROMPT_DICT = {
    "prompt_input": (
        "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"

    ),
    "prompt_no_input": (
        "以下に、あるタスクを説明する指示があります。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    ),
}

def create_prompt(instruction, input=None):
    """
    Generates a prompt based on the given instruction and an optional input.
    If input is provided, it uses the 'prompt_input' template from PROMPT_DICT.
    If no input is provided, it uses the 'prompt_no_input' template.

    Args:
        instruction (str): The instruction describing the task.
        input (str, optional): Additional input providing context for the task. Default is None.

    Returns:
        str: The generated prompt.
    """
    if input:
        # Use the 'prompt_input' template when additional input is provided
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        # Use the 'prompt_no_input' template when no additional input is provided
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    
# Example usage
instruction_example = "以下のトピックに関する詳細な情報を提供してください。"
input_example = "東京工業大学の主なキャンパスについて教えてください"
prompt = create_prompt(instruction_example, input_example)

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
pprint(out)