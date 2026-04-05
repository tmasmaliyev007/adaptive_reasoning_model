import requests
from typing import List

class Tokenizer:
    def __init__(self, base_url: str):
        self.url = f"{base_url}/tokenize"
    
    def encode(self, text: str) -> List[int]:
        response = requests.post(self.url, json={"content": text})
        return response.json()["tokens"]
    
    def count(self, text: str) -> int:
        return len(self.encode(text))