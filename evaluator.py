from benchmark.commonsense_qa import evaluate_commonsense_qa

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

if __name__ == '__main__':
    model_name = "model/gguf/qwen-3-4B/b16-full.gguf"
    chat_template_name = "template/qwen3.jinja"
    MAX_SEQ_LENGTH = 20000

    # Load the model
    llm = Llama(
        model_path = model_name,
        n_gpu_layers = 0,
        verbose = False,
        flash_attn = True,
        n_ctx = MAX_SEQ_LENGTH
    )

    try:
        # Load Template
        with open(chat_template_name, 'r') as f:
            chat_template = f.read()

        # Load formatter
        formatter = Jinja2ChatFormatter(
            template=chat_template,
            add_generation_prompt=True,
            bos_token=llm._model.token_get_text(llm.token_bos()),
            eos_token=llm._model.token_get_text(llm.token_eos())
        )

        evaluate_commonsense_qa(
            model = llm,
            formatter = formatter,
            limit = None,
            max_new_tokens = MAX_SEQ_LENGTH,
            top_k = 40,
            temperature = 0.7,
            top_p = 1,
            repeat_penalty = 1.0,
            output_dir = "results/commonsense_qa/exp1"
        )
    finally:
        llm.close()