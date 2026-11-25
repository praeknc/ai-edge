"""Entry point for Streamlit Community Cloud (keeps requirements.txt and packages.txt next to it).

Deploy with main file path: host-copilot/streamlit_app.py
Locally you can run either this file or app/frontend.py.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "frontend.py"), run_name="__main__")
