LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}"""

GEMMA_CHAT_TEMPLATE = """<bos><start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{response}"""


def gemma_chat_template_fn(contents):
    prompt = contents[0]['content']
    response = contents[1]['content'] if len(contents) >= 2 else ''
    return GEMMA_CHAT_TEMPLATE.format(instruction=prompt, response=response)

def llama_chat_template_fn(contents):
    prompt = contents[0]['content']
    response = contents[1]['content'] if len(contents) >= 2 else ''
    return LLAMA3_CHAT_TEMPLATE.format(instruction=prompt, response=response)
    
model_chat_template_fn_dict = {
    'google/gemma-2-2b-it': gemma_chat_template_fn,
    'google/gemma-2-9b-it': gemma_chat_template_fn,
    'meta-llama/Llama-3.2-3B-Instruct': llama_chat_template_fn,
}

def get_chat_template_fn(model_path):
    return model_chat_template_fn_dict[model_path]