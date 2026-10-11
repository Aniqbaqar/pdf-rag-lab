# PDF RAG Lab (Free-tier setup)

A beginner-friendly Retrieval-Augmented Generation (RAG) demo for an AI lab.

## Files
- `app.py` — Streamlit front end.
- `rag_engine.py` — PDF reading, chunking, embeddings, vector search, and LLM call.
- `requirements.txt` — dependencies.

## Important about “free unlimited tokens”
This project does not require a paid subscription, but **no hosted free LLM service guarantees unlimited tokens**. The Groq developer/free tier can have rate limits and model availability can change. Check the current model list and usage limits in the provider dashboard. The model ID in the app can be changed in the UI.

## Run locally
Use Python 3.10–3.12 if possible.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal. Get a Groq API key from the Groq developer console and paste it into the app. Never commit your API key to GitHub.

## Upload to GitHub
1. Sign in at https://github.com.
2. Click **New repository** and name it `pdf-rag-lab`.
3. Choose Public or Private, then create the repository.
4. Click **Add file → Upload files**.
5. Upload `app.py`, `rag_engine.py`, `requirements.txt`, and `README.md`.
6. Click **Commit changes**.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io and sign in using GitHub.
2. Click **Create app**.
3. Select your repository (`pdf-rag-lab`), branch (`main`), and main file path (`app.py`).
4. Open **Advanced settings** if you want to select a Python version; Python 3.12 is a reasonable starting point.
5. Click **Deploy** and wait for the build to finish.
6. Open the deployed app URL and test with a small text-based PDF.
7. Enter your Groq API key into the app UI when using it. It is not saved in the app's code or session state.

## How to teach the lab
1. Upload a PDF with selectable text.
2. Click **Process PDF / Build RAG index**.
3. Ask a question that is answered in the PDF.
4. Open **View retrieved PDF passages** and compare the retrieved text with the generated answer.
5. Ask something not covered by the PDF and observe whether the model admits the information is missing.

## Notes / troubleshooting
- **Scanned PDF:** `pypdf` extracts selectable text, not text embedded only as images. OCR is needed for scanned PDFs.
- **First run is slower:** `sentence-transformers/all-MiniLM-L6-v2` downloads on first use.
- **Build fails on a small hosted instance:** `sentence-transformers` and `faiss-cpu` use memory. Try a smaller PDF or run locally for the classroom.
- **API error:** check the API key, current supported model ID, provider rate limits, and account eligibility.
- **Not model training:** RAG does not train/fine-tune the LLM. It builds a searchable index from the PDF and supplies relevant passages at question time.
"# pdf-rag-lab" 
