import json
import os

from openai import OpenAI
from src.constants.paths import SECRET_PATH, URL_ONE_API
import logging


class PortailIAG:
    def __init__(self):
        os.environ["NO_PROXY"] = "oneapi.edf.fr"
        os.environ["no_proxy"] = "oneapi.edf.fr"

        secret = self._load_secret()
        self.project_id = secret["ID_PROJET"]
        self.api_key = secret["API_KEY"]
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=URL_ONE_API,
            default_headers={"Accept": "application/octet-stream"},
        )

    def _load_secret(self):
        with open(SECRET_PATH) as f:
            return json.load(f)["PORTAIL_IAG"]

    # ✅ Gemini avec response_format
    def query_gemini(self, model: str, prompt: str, query: str, response_format: dict):
        logger.info("Calling Gemini with response_format")
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            timeout=600.0,
            temperature=0.0,
            max_completion_tokens=32000,
            response_format=response_format,
        )

    # ✅ Mistral sans response_format
    def query_mistral(
        self, model: str, prompt: str, query: str = None, streaming: bool = True
    ):
        if query:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]
        else:
            messages = [
                {"role": "system", "content": prompt},
            ]

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=600.0,
            temperature=0.0,
            max_tokens=16192,
            stream=streaming,
        )

        headers = stream.response.headers
        correlation_id = (
            headers.get("x-correlation-id")
            or headers.get("x-request-id")
            or headers.get("request-id")
            or headers.get("cf-ray")
        )

        return list(stream), correlation_id

    def query_mistral_json_structured(
        self,
        model: str,
        system_prompt: str,
        json_schema,
        max_tokens: int = 16192,
    ):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            timeout=300.0,
        )

        # With structured output, content is already valid JSON
        return completion.choices[0].message.content
