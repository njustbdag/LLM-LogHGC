import requests
import pandas as pd
import time


class FlexibleLogTemplateExtractor:
    # "deepseek-r1:14b", "qwen3:14b"
    def __init__(self, model="qwen3:14b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []

    def check_connection(self):
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def extract_template(self, log_message: str, custom_prompt: str = None) -> str:
        if not self.check_connection():
            raise ConnectionError("can not link Ollama.")
        prompt = self._build_custom_prompt(log_message, custom_prompt)
        response = self._call_model(prompt)
        template = self._clean_template_response(response)
        result = {
            "original_log": log_message,
            "template": template,
            "custom_prompt": custom_prompt if custom_prompt else "other prompt",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.results.append(result)
        return template

    def _build_custom_prompt(self, log_message: str, custom_prompt: str) -> str:
        prompt = f"""
        {custom_prompt}
        """
        return prompt

    def _call_model(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 65536, "top_p": 0.9},
        }
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"API call failed: {response.status_code}")

    def _clean_template_response(self, response: str) -> str:
        cleaned = response.strip()
        cleaned = cleaned.strip("\"'")
        return cleaned
