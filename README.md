## InsightX Conversational Analytics Prototype

This prototype implements the end-to-end pipeline you described using **LangChain** and **Gemini** on top of your synthetic digital payments dataset.

### High-level architecture

- **Input**: User's natural language question.
- **Context**: The `schema.txt` description of the transactions dataset.
- **LLM 1 – Task planner** (`plan_tasks` in `insight_chain.py`):
  - Reads the schema description.
  - Decomposes the question into 1–4 analysis tasks.
  - Generates a **`SELECT`-only SQL query** per task over a DuckDB table named `transactions`.
- **DB layer** (`execute_tasks` in `insight_chain.py`):
  - Loads `dataset.csv` into DuckDB.
  - Executes each LLM-generated SQL query.
  - Produces a **unified JSON structure**:
    ```json
    {
      "tasks": [
        {
          "task_name": "...",
          "task_description": "...",
          "sql_query": "SELECT ...",
          "rows": [ { "col": "value", "...": "..." } ]
        }
      ]
    }
    ```
- **LLM 2 – Answer generator** (`generate_answer` in `insight_chain.py`):
  - Consumes the original question + unified JSON.
  - Produces an **explainable, leadership-friendly natural language answer**, as expected in the challenge brief.
- **UI layer** (`app.py`, Streamlit):
  - Text box for the question.
  - Shows:
    - **Natural language answer**.
    - **Visuals** (simple auto-inferred Plotly chart + table for the first task).
    - **Raw response JSON** (ready to be fed to any custom graph library you prefer).

### Setup

1. **Create and populate `.env`**

   Copy the example and fill in your Gemini key:

   ```bash
   cp .env.example .env
   # then edit .env and set GOOGLE_API_KEY
   ```

2. **Create a virtual environment (recommended) and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on macOS / Linux
   # .venv\\Scripts\\activate  # on Windows

   pip install -r requirements.txt
   ```

3. **Ensure data files are present**

   - `schema.txt` – challenge description + schema (already provided).
   - `dataset.csv` – the 100-row synthetic dataset we created earlier.

### Run the chat interface

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`) in your browser.

### Notes

<!-- - The LLM model name defaults to `gemini-3-pro-preview`, but you can override it by setting `GEMINI_MODEL_NAME` in `.env`. -->
- The SQL guardrail is intentionally strict: only `SELECT` queries are allowed; anything else is strictly forbidden and returns an error row in the JSON.
- The visualization logic in `app.py` now renders **per-task Plotly charts** driven directly from the `response_json` structure:
  - Line or bar charts over `hour_of_day` and `day_of_week` when present.
  - Categorical vs metric bar charts for segments like `transaction_type`, `device_type`, `network_type`, `sender_bank`, or `merchant_category`.
  - A tabbed layout that shows one chart + table per analysis task, making it easy to plug in or mirror the same JSON into any external JS graph library if needed.

## Prototype
Prototype for testing purposes is deployed at: [https://techsas-analytics.streamlit.app](https://techsas-analytics.streamlit.app)