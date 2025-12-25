from typing import Optional
import os


# =====================================================
# v3.5 LLM CLIENT WRAPPER
# =====================================================
# This module is intentionally thin.
# It must NOT contain business logic.
# =====================================================


class LLMDisabledError(RuntimeError):
    """Raised when LLM usage is disabled by config."""


class LLMConfigurationError(RuntimeError):
    """Raised when LLM is enabled but not configured properly."""


class LLMClient:
    """
    Provider-agnostic LLM client.

    Responsibilities:
    - Respect config flags
    - Call external LLM
    - Return raw text only

    Non-responsibilities:
    - Prompt design
    - Insight logic
    - Validation
    """

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.provider = config.get("provider", "openai")
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 300)

        if self.enabled:
            self._validate_config()

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate text from the LLM.

        Input:
        - prompt: fully constructed prompt string

        Output:
        - plain text response
        """

        if not self.enabled:
            raise LLMDisabledError(
                "LLM narrative is disabled. Enable it via config."
            )

        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if self.provider == "openai":
            return self._call_openai(prompt)

        raise LLMConfigurationError(
            f"Unsupported LLM provider: {self.provider}"
        )

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _validate_config(self):
        if not self.model:
            raise LLMConfigurationError(
                "LLM model must be specified when narrative is enabled"
            )

        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMConfigurationError(
                    "OPENAI_API_KEY environment variable is missing"
                )

    def _call_openai(self, prompt: str) -> str:
        """
        OpenAI implementation (isolated).
        """

        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMConfigurationError(
                "openai package is not installed"
            ) from e

        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional business analyst assistant. "
                        "You must strictly follow instructions in the user prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response or not response.choices:
            raise RuntimeError("Empty response from LLM")

        content: Optional[str] = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content")

        return content.strip()

