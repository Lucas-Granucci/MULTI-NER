import os
from rich.text import Text
from rich.console import Console

console = Console()

def print_message(message: str, style: str = "bold blue"):
    width = os.get_terminal_size().columns
    line_length = ((width - len(message)) // 2) - 2
    message = f"#{'-'*line_length} {message} {'-'*line_length}#"
    console.print(Text(message, style=style))

def print_submessage(message: str, style: str = "blue"):
    console.print(Text(message, style=style))