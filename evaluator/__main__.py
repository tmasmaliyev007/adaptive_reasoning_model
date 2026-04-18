from evaluator.runner import evaluate_dataset

from llama_client.tokenizer import Tokenizer
from openai import AsyncOpenAI
import asyncio
import requests
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess

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

    max_seq_length = cfg.model.seq_length_for_slot * cfg.concurrent_calls
    try:
        server = subprocess.Popen(
            [
                "llama-server",
                "-m",                   cfg.model.path,
                "--chat-template-file", cfg.model.chat_template,
                "-c",                   str(max_seq_length),
                "--host",               str(cfg.connection.host),
                "--port",               str(cfg.connection.port),
                "--special",
                "-ngl",                 str(cfg.load_num_of_gpu_layers),
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
            filepath       = cfg.dataset.path,
            limit          = None if cfg.model.limit == -1 else cfg.model.limit,
            max_new_tokens = max_seq_length,
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

@hydra.main(config_path="configs", config_name="config", version_base=None)
def initialize_and_run(cfg: DictConfig) -> None:
    asyncio.run(main(cfg))

if __name__ == '__main__':
    initialize_and_run()