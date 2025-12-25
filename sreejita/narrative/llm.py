from typing import Optional
import os


class LLMDisabledError(RuntimeError):
    pass


class LLMConfigurationError(RuntimeError):
    pass


class LLMClient:
    """
    Provider-agnostic LLM client (v3.5)

    Supported providers:
    - openai
    - gemini
    """

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.provider = config.get("provider", "openai")
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 300)

        if self.enabled:
            self._validate_config()

    # -----------------------------
    # PUBLIC
    # -----------------------------

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            raise LLMDisabledError("LLM narrative is disabled")

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if self.provider == "openai":
            return self._call_openai(prompt)

        if self.provider == "gemini":
            return self._call_gemini(prompt)

        raise LLMConfigurationError(f"Unsupported provider: {self.provider}")

    # -----------------------------
    # VALIDATION
    # -----------------------------

    def _validate_config(self):
        if not self.model:
            raise LLMConfigurationError("LLM model must be specified")

        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMConfigurationError("OPENAI_API_KEY is missing")

        if self.provider == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise LLMConfigurationError("GEMINI_API_KEY is missing")

    # -----------------------------
    # PROVIDERS
    # -----------------------------

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMConfigurationError("openai package not installed") from e

        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional business narrator."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content: Optional[str] = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from OpenAI")

        return content.strip()

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise LLMConfigurationError(
                "google-generativeai package not installed"
            ) from e

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel(self.model)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        if not response or not response.text:
            raise RuntimeError("Empty response from Gemini")

        return response.text.strip()
