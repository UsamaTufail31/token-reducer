"""Tokenizer abstraction layer for multiple LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class TokenizerInterface(ABC):
    """Abstract base class for tokenizer implementations."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get tokenizer name."""
        pass


class TikTokenTokenizer(TokenizerInterface):
    """OpenAI tokenizer using tiktoken."""

    def __init__(self, model_name: str = "gpt-4") -> None:
        """Initialize tiktoken tokenizer.

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')

        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If model is not supported
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for OpenAI tokenizers. "
                "Install with: pip install tiktoken"
            )

        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
            self._model_name = model_name
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._model_name = f"{model_name} (using cl100k_base)"

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._encoding.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))

    @property
    def name(self) -> str:
        """Get tokenizer name."""
        return f"tiktoken:{self._model_name}"


class HuggingFaceTokenizer(TokenizerInterface):
    """HuggingFace tokenizer wrapper."""

    def __init__(self, model_name: str = "gpt2") -> None:
        """Initialize HuggingFace tokenizer.

        Args:
            model_name: HuggingFace model name or path

        Raises:
            ImportError: If transformers is not installed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for HuggingFace tokenizers. "
                "Install with: pip install transformers torch"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model_name = model_name

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))

    @property
    def name(self) -> str:
        """Get tokenizer name."""
        return f"huggingface:{self._model_name}"


class AnthropicTokenizer(TokenizerInterface):
    """Anthropic tokenizer wrapper."""

    def __init__(self, model_name: str = "claude-3-opus-20240229") -> None:
        """Initialize Anthropic tokenizer.

        Args:
            model_name: Anthropic model name

        Raises:
            ImportError: If anthropic is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic is required for Anthropic tokenizers. "
                "Install with: pip install anthropic"
            )

        self._client = Anthropic()
        self._model_name = model_name

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Note: Anthropic API doesn't expose token IDs directly.
        This returns a placeholder list with length equal to token count.
        """
        count = self.count_tokens(text)
        return list(range(count))  # Placeholder

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Note: Not supported by Anthropic API.
        """
        raise NotImplementedError(
            "Anthropic tokenizer does not support decoding token IDs"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Anthropic's API."""
        # Use the count_tokens method from Anthropic SDK
        result = self._client.count_tokens(text)
        return result

    @property
    def name(self) -> str:
        """Get tokenizer name."""
        return f"anthropic:{self._model_name}"


class CharacterTokenizer(TokenizerInterface):
    """Simple character-based tokenizer (fallback)."""

    def __init__(self, chars_per_token: int = 4) -> None:
        """Initialize character tokenizer.

        Args:
            chars_per_token: Average characters per token (default: 4)
        """
        self._chars_per_token = chars_per_token

    def encode(self, text: str) -> List[int]:
        """Encode text to character indices."""
        return [ord(c) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode character indices to text."""
        return "".join(chr(i) for i in token_ids)

    def count_tokens(self, text: str) -> int:
        """Estimate token count based on character count."""
        return max(1, len(text) // self._chars_per_token)

    @property
    def name(self) -> str:
        """Get tokenizer name."""
        return f"character:{self._chars_per_token}"


def create_tokenizer(
    tokenizer_name: str, model_name: Optional[str] = None
) -> TokenizerInterface:
    """Factory function to create appropriate tokenizer.

    Args:
        tokenizer_name: Tokenizer type ('tiktoken', 'huggingface', 'anthropic', 'character')
                       or model name (e.g., 'gpt-4', 'claude-3')
        model_name: Optional model name override

    Returns:
        TokenizerInterface implementation

    Raises:
        ValueError: If tokenizer type is not supported
        ImportError: If required library is not installed

    Examples:
        >>> tokenizer = create_tokenizer('gpt-4')
        >>> tokenizer = create_tokenizer('tiktoken', 'gpt-3.5-turbo')
        >>> tokenizer = create_tokenizer('huggingface', 'bert-base-uncased')
    """
    # Auto-detect from model name
    tokenizer_lower = tokenizer_name.lower()

    # OpenAI models
    if any(x in tokenizer_lower for x in ["gpt-", "text-davinci", "text-embedding"]):
        return TikTokenTokenizer(model_name or tokenizer_name)

    # Anthropic models
    if "claude" in tokenizer_lower:
        return AnthropicTokenizer(model_name or tokenizer_name)

    # Explicit tokenizer types
    if tokenizer_lower == "tiktoken":
        return TikTokenTokenizer(model_name or "gpt-4")

    if tokenizer_lower == "huggingface" or tokenizer_lower == "hf":
        return HuggingFaceTokenizer(model_name or "gpt2")

    if tokenizer_lower == "anthropic":
        return AnthropicTokenizer(model_name or "claude-3-opus-20240229")

    if tokenizer_lower == "character" or tokenizer_lower == "char":
        return CharacterTokenizer()

    # Try HuggingFace as fallback (most flexible)
    if TRANSFORMERS_AVAILABLE:
        try:
            return HuggingFaceTokenizer(tokenizer_name)
        except Exception:
            pass

    # Final fallback to character-based
    return CharacterTokenizer()
