import logging
from rich.logging import RichHandler
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set default logging level
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler()]
)

logger = logging.getLogger('rich')

def print_message(message: str, style: str = "bold yellow"):
    styled_message = Text(f"# {message} #", style=style)
    logger.info(styled_message)

def print_submessage(message: str, style: str = "dim"):
    styled_message = Text(message, style=style)
    logger.info(styled_message)