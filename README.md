TDS Project 2 - May 2025

This project implements a FastAPI-based data analysis agent powered by Groq's LLaMA models.
It accepts a questions.txt file and optional dataset files, generates a plan, runs Python analysis code inside a Docker container, and returns the results in the requested JSON format.
Features

    Handles file uploads (questions.txt + any CSV, image, etc.)
    Uses Groq API for planning and assembling final results
    Executes Python analysis code securely in a sandboxed Docker container (python:3.11)
    Supports web scraping, data processing, and plotting (Base64-encoded PNGs)
    Fully CORS-enabled

Requirements

    Python 3.11+
    Docker installed and running
    Groq API key

Installation

Clone or download this repository.

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies: pip install -r requirements.txt

    Create a .env file with: GEMINI_API_KEY=your_api_key_here

    Run the app: uvicorn app:app --host 0.0.0.0 --port 8000

Usage
-POST to /api/ with multipart/form-data: -questions.txt (required) -Additional files (optional) -Response will be JSON or Base64-encoded image depending on the query.