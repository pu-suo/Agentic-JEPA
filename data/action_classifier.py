import re

# Patterns that indicate external/environment-query actions
EXTERNAL_PATTERNS = [
    r'requests\.',           # HTTP requests
    r'urllib\.',              # URL lib
    r'subprocess\.',          # Shell commands
    r'subprocess\.run',
    r'subprocess\.call',
    r'subprocess\.Popen',
    r'os\.system\(',          # OS calls
    r'os\.popen\(',
    r'os\.remove\(',          # File deletion
    r'os\.unlink\(',
    r'shutil\.',              # File operations
    r'pip\s+install',         # Package installation
    r'pip3\s+install',
    r'apt[\s-]',              # System packages
    r'apt-get',
    r'wget\s',               # Downloads
    r'curl\s',
    r'open\(.+["\']https?:', # URL file opens
    r'socket\.',              # Network sockets
    r'selenium\.',            # Browser automation
    r'playwright\.',
    r'sqlite3\.connect',      # Database connections
    r'psycopg2\.',
    r'pymysql\.',
    r'sqlalchemy\.',
    r'open\([^)]+,\s*["\']w', # File writes
    r'open\([^)]+,\s*["\']a', # File appends
    r'with\s+open\(.+["\']w', # Context manager file writes
    r'Path\(.+\)\.write',     # pathlib writes
    r'\.to_csv\(',            # DataFrame file exports
    r'\.to_excel\(',
    r'json\.dump\(',          # JSON file writes (to file handle)
    r'!pip\s',                # Jupyter shell commands
    r'!apt\s',
    r'!wget\s',
    r'!curl\s',
    r'get_ipython\(\)',       # Jupyter system calls
    r'import\s+paramiko',     # SSH
    r'import\s+ftplib',       # FTP
    r'import\s+smtplib',      # Email
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
