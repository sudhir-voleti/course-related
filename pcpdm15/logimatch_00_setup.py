# logimatch_00_setup.py
# Cell 1: Quiet Package Installation & Core Imports for LogiMatch Pro Workshop

import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

PACKAGES = [
    "google-genai",
    "beautifulsoup4",
    "requests",
    "pydantic",
    "ipywidgets",
    "ddgs",
]

print("⏳ Installing required libraries quietly...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", *PACKAGES],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Literal

import ipywidgets as widgets
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from google import genai
from google.genai import types
from IPython.display import HTML, clear_output, display
from pydantic import BaseModel, Field

print("✅ Setup complete: All libraries (google-genai, bs4, ddgs, pydantic, ipywidgets) loaded.")
