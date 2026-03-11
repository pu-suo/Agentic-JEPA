import re

# Patterns that indicate external/environment-query actions
EXTERNAL_PATTERNS = [
    r'requests\.',           # HTTP requests
    r'urllib\.',              # URL lib
    r'subprocess\.',          # Shell commands
    r'os\.system\(',          # OS calls
    r'os\.popen\(',
    r'pip\s+install',         # Package installation
    r'apt[\s-]',              # System packages
    r'wget\s',               # Downloads
    r'curl\s',
    r'open\(.+["\']https?:', # URL file opens
    r'socket\.',              # Network sockets
    r'selenium\.',            # Browser automation
    r'playwright\.',
]

EXTERNAL_RE = re.compile('|'.join(EXTERNAL_PATTERNS), re.IGNORECASE)


def classify_action(action_text: str) -> float:
    """
    Returns τ(a_t): the JEPA loss weight for this action.

    - Internal actions (deterministic, self-contained): τ → 1.0
    - External actions (environment-query, stochastic): τ → 0.1

    This is a hardcoded heuristic for the prototype.
    In later phases, replace with a learned classifier trained on
    afterstate target variance.
    """
    if EXTERNAL_RE.search(action_text):
        return 0.1  # Low JEPA weight: afterstate is semantically thin
    return 1.0      # Full JEPA weight: afterstate is deterministic and learnable
