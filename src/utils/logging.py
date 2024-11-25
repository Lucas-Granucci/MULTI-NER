import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set default logging level
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("rich")
