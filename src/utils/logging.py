from rich.console import Console
from rich.text import Text

console = Console()

def print_message(message: str, style: str = "bold blue"):
    message = f"#{'='*60} {message} {'='*60}#"
    console.print(Text(message, style=style))

def print_submessage(message: str, style: str = "blue"):
    console.print(Text(message, style=style))