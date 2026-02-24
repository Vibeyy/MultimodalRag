"""Script to start Streamlit UI."""

import subprocess
import sys
from pathlib import Path

from multimodal_rag.utils.logger import setup_logger

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting Streamlit UI...")
    
    # Use the lazy-loading version (works best)
    app_path = Path(__file__).parent.parent / "streamlit_lazy.py"
    
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--theme.base=light",
    ])
