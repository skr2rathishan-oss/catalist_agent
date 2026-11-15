"""Launch the local Gradio test UI or the configured Hugging Face Space."""

import os

from app.ui import build_ui


if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    build_ui().queue(default_concurrency_limit=1, max_size=8).launch(server_name=host, share=False, show_error=False)
