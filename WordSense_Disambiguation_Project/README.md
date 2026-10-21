# WordSense AI Lab — Word Sense Disambiguation

A colorful Streamlit web app for the lab question: develop a Word Sense Disambiguation (WSD) / Word Sense Induction system using WordNet and lexical relations.

## Features
- WordNet sense inventory for ambiguous words.
- Definitions, examples, synonyms, hypernyms, and hyponyms.
- Simplified Lesk WSD prediction for a sentence.
- Downloadable sense inventory and WSD report.
- Short analysis section for the lab deliverables.
- Colorful gradient-based frontend.

## Run in VS Code (Windows)
1. Extract this ZIP file.
2. Open the extracted `WordSense_Disambiguation_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the app:
   ```powershell
   streamlit run app.py
   ```
7. A browser tab should open at `http://localhost:8501`.

## First run
The app attempts to download NLTK WordNet and tokenizer resources automatically. Internet access may be needed the first time. Keep the terminal open while using the app.

## Quick demo
- Sense Explorer: search `bank`, `bat`, `light`, or `plant`.
- Disambiguation: use a sentence such as `She deposited money at the bank.` and target word `bank`.
- Download the generated text reports from the app.

## Algorithm note
The WSD tab uses NLTK's Simplified Lesk algorithm. This is a simple baseline, not a clinically or universally accurate language model. It may select an incorrect sense if the context does not overlap well with WordNet glosses/examples.
