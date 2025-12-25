import os
from typing import Optional


class LLMDisabledError(RuntimeError):
    pass


class LLMConfigurationError(RuntimeError):
    pass


class LLMRuntimeError(RuntimeError):
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

        # 🔐 Safe defaults
        if self.provider == "gemini":
            self.model = config.get(
                "model", "gemini-1.5-flash-latest"
            )
        else:
            self.model = config.get(
                "model", "gpt-4o-mini"
            )

        self.temperature = float(config.get("temperature", 0.2))
        self.max_tokens = int(config.get("max_tokens", 300))

        if self.enabled:
            self._validate_config()

    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------

    def generate(self, prompt: str) -> str:
        if not self.enabled:
            raise LLMDisabledError("LLM narrative is disabled")

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        try:
            if self.provider == "openai":
                return self._call_openai(prompt)

            if self.provider == "gemini":
                return self._call_gemini(prompt)

        except Exception as e:
            raise LLMRuntimeError(
                f"{self.provider.upper()} LLM failed: {str(e)}"
            ) from e

        raise LLMConfigurationError(
            f"Unsupported LLM provider: {self.provider}"
        )

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    def _validate_config(self):
        if not self.model:
            raise LLMConfigurationError("LLM model must be specified")

        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMConfigurationError("OPENAI_API_KEY is missing")

        if self.provider == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise LLMConfigurationError("GEMINI_API_KEY is missing")

    # -------------------------------------------------
    # PROVIDERS
    # -------------------------------------------------

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMConfigurationError(
                "openai package not installed"
            ) from e

        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional business analyst summarizing insights.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content: Optional[str] = response.choices[0].message.content
        if not content:
            raise LLMRuntimeError("Empty response from OpenAI")

        return content.strip()

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise LLMConfigurationError(
                "google-generativeai package not installed"
            ) from e

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        try:
            model = genai.GenerativeModel(self.model)

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )

        except Exception as e:
            raise LLMRuntimeError(
                "Gemini API error. "
                "Check model name, API key, and access permissions."
            ) from e

        if not response or not getattr(response, "text", None):
            raise LLMRuntimeError("Empty response from Gemini")

        return response.text.strip()
