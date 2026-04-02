from benchmark.commonsense_qa import evaluate_commonsense_qa

from openai import OpenAI
import requests
import time

import argparse
import subprocess
import os

HOST = "127.0.0.1"
PORT = "8080"
BASE_URL = f"http://{HOST}:{PORT}"
NUMBER_GPU_LAYER = 0

def wait_for_server(url: str, timeout: int = 120, intervals: int = 2) -> bool:
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

def main(args):
    model_path = args.model_path
    if not os.path.isfile(model_path):
        from huggingface_hub import snapshot_download
        model_dir = "model"
        print("The file was not found in the directory. Now, considering input as huggingface repository and is automatically being searched...")

        snapshot_download(
            repo_id=args.model_path,
            local_dir=model_dir,
            allow_patterns=["*.gguf"]
        )

        filename = os.listdir(model_dir)
        if not filename:
            raise ValueError(f"There are no models inside repository with .gguf extension")

        model_path = f"{model_dir}/{filename}"

    try:
        server = subprocess.Popen(
            [
                "llama-server",
                "-m", args.model_path,
                "--chat-template-file", args.chat_template_path,
                "-c", str(args.max_seq_length),
                "--host", HOST,
                "--port", PORT,
                "--special",
                "-ngl", f"{NUMBER_GPU_LAYER}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        wait_for_server(
            url = f"{BASE_URL}/health",
            timeout = 120,
            intervals = 2
        )

        client = OpenAI(
            base_url = BASE_URL,
            api_key = "none"
        )

        evaluate_commonsense_qa(
            client = client,
            limit = None if args.limit == -1 else args.limit,
            max_new_tokens = args.max_seq_length,
            temperature = args.temperature,
            top_k = args.top_k,
            top_p = args.top_p,
            repeat_penalty = args.repeat_penalty,
            output_dir = args.experiment_dir
        )
    except Exception as e:
        print(e)
    finally:
        if server is not None:
            server.terminate()
            server.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',         type=str,    required=True)
    parser.add_argument('--chat-template-path', type=str,    required=True)
    parser.add_argument('--max-seq-length',     type=int,    required=True)

    parser.add_argument('--limit',              type=int,    default=-1)
    parser.add_argument('--temperature',        type=float,  default=1.0)
    parser.add_argument('--top-k',              type=int,    default=40)
    parser.add_argument('--top-p',              type=float,  default=1.0)
    parser.add_argument('--repeat-penalty',     type=float,  default=1.0)
    parser.add_argument('--experiment-dir',     type=str,    default=None)

    args = parser.parse_args()

    main(args)