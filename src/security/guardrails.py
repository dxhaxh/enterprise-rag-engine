import re

class SecurityGuardrails:
    """
    Enterprise firewall for LLM inputs and outputs.
    Prevents prompt injection and accidental data leakage.
    """
    
    # 1. Blacklist for Prompt Injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"forget\s+all\s+previous",
        r"system\s+prompt",
        r"bypass\s+security",
        r"you\s+are\s+now\s+unbound",
        r"act\s+as\s+(an\s+)?uncensored"
    ]

    # 2. Blacklist for Data Leakage (e.g., AWS keys, internal IP addresses)
    # We use a simple regex to catch anything that looks like a secret API key or internal IP
    SENSITIVE_PATTERNS = [
        r"sk-[a-zA-Z0-9]{32,}",                # Matches OpenAI-style API keys
        r"AKIA[0-9A-Z]{16}",                  # Matches AWS Access Keys
        r"\b192\.168\.\d{1,3}\.\d{1,3}\b"     # Matches Internal network IPs
    ]

    @classmethod
    def validate_input(cls, query: str) -> bool:
        """Returns False if a malicious prompt injection is detected."""
        query_lower = query.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return False  # Violation found!
        return True

    @classmethod
    def validate_output(cls, answer: str) -> bool:
        """Returns False if the LLM attempts to leak sensitive data."""
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, answer):
                return False  # Data leak found!
        return True