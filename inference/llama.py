from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

def run_llama_inference(
    model: Llama,
    instruction: str,
    formatter: Jinja2ChatFormatter,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.0,
) -> str:

    messages = [{"role": "user", "content": instruction}]
    formatted_prompt = formatter(messages=messages).prompt

    tokens = model.tokenize(
        formatted_prompt.encode("utf-8"),
        add_bos=False,       
        special=True
    )

    # Reset KV Cache
    model.reset()

    # Evaluate tokens & Store KV Cache
    model.eval(tokens)

    output_tokens = []

    # Autoregressive decoder
    for _ in range(max_new_tokens):
        token = model.sample(
            top_k = top_k,
            top_p = top_p,
            temp = temperature,
            repeat_penalty = repeat_penalty,
        )

        if token == model.token_eos():
            break

        output_tokens.append(token)
        
        # Add new token & 
        model.eval([token])
    
    output_parts = []
    for token in output_tokens:
        # Tokenize the output tokens one by one
        piece = model.detokenize([token], special=True)
        output_parts.append(piece.decode("utf-8", errors="replace"))

    return "".join(output_parts)