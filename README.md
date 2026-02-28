# AB's Transformation Journey - Weight Tracker

A Streamlit dashboard that visualises daily and weekly weight progress from a Google Sheet.

---

## Dev Workflow

### First time setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Making changes

1. Activate the virtual environment (if not already active):
   ```bash
   source .venv/bin/activate
   ```

2. Run the app locally:
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501` and **auto-reloads** on every file save.

3. Edit `app.py` and save — the browser refreshes automatically.

4. When happy with the changes, commit and push:
   ```bash
   git add app.py
   git commit -m "describe your change"
   git push
   ```

5. Streamlit Cloud picks up the push and **redeploys automatically** within ~1 minute.

---

## Project Structure

```
weight_tracker/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Key Details

| Item | Value |
|---|---|
| Data source | Google Sheet (published as CSV) |
| GitHub repo | https://github.com/ashutoshbhardwaj/weight_tracker |
| Hosted on | Streamlit Community Cloud |
| Password | Set in Streamlit Cloud → App Settings → Secrets |

## Shareable Link

Append `?pwd=yourpassword` to the app URL to share a link that skips the password prompt:

```
https://<your-app>.streamlit.app/?pwd=yourpassword
```
