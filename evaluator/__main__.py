from evaluator.runner import evaluate_dataset

from llama_client.tokenizer import Tokenizer
from openai import AsyncOpenAI
import asyncio
import requests
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess
import os

import logging

logger = logging.getLogger(__name__)

def wait_for_server(url: str, timeout: int, intervals: int) -> bool:
    start = time.time()
    print("Waiting for llama-server...")

    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            data = r.json()
            if data.get("status") == "ok":
                print("Server is ready")
                return True
            
        except (requests.ConnectionError, requests.Timeout):
            pass
    
        time.sleep(intervals)
    
    raise TimeoutError(f"Server not ready after {timeout}s")

async def main(cfg: DictConfig):
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = OmegaConf.create(container)

    model_path = cfg.model.path

    if not os.path.isfile(model_path):
        from huggingface_hub import snapshot_download
        model_dir = "model"
        print("The file was not found in the directory. Now, considering input as huggingface repository and is automatically being searched...")

        snapshot_download(
            repo_id        = cfg.model.huggingface_src,
            local_dir      = model_dir,
            allow_patterns = ["*.gguf"]
        )

        filename = os.listdir(model_dir)
        if not filename:
            raise ValueError(f"There are no models inside repository with .gguf extension")

        model_path = f"{model_dir}/{filename[0]}"

    try:
        server = subprocess.Popen(
            [
                "llama-server",
                "-m",                   model_path,
                "--chat-template-file", cfg.chat_template_path,
                "-c",                   str(cfg.model.max_seq_length),
                "--host",               str(cfg.connection.host),
                "--port",               str(cfg.connection.port),
                "--special",
                "-ngl",                 str(cfg.model.load_num_of_gpu_layers),
                "-np",                  str(cfg.concurrent_calls)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        wait_for_server(
            url       = cfg.connection.health,
            timeout   = cfg.connection.timeout,
            intervals = cfg.connection.interval
        )

        client = AsyncOpenAI(
            base_url = cfg.connection.url,
            api_key = "none"
        )

        tokenizer = Tokenizer(base_url = cfg.connection.url)

        await evaluate_dataset(
            client         = client,
            tokenizer      = tokenizer,
            filepath       = cfg.data_path,
            limit          = None if cfg.model.limit == -1 else cfg.model.limit,
            max_new_tokens = cfg.model.max_seq_length,
            temperature    = cfg.model.temperature,
            top_k          = cfg.model.top_k,
            top_p          = cfg.model.top_p,
            repeat_penalty = cfg.model.repeat_penalty,
            output_dir     = cfg.experiment_dir,
            concurrency    = cfg.concurrent_calls
        )
    except Exception as e:
        print(e)
    finally:
        await client.close()

        if server is not None:
            server.terminate()
            server.wait()

@hydra.main(config_path="evaluator/configs", config_name="config")
def get_config(cfg: DictConfig) -> DictConfig:
    return cfg

if __name__ == '__main__':
    cfg = get_config()

    asyncio.run(main(cfg))