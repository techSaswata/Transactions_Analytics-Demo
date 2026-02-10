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

- The LLM model name defaults to `gemini-3-pro-preview`, but you can override it by setting `GEMINI_MODEL_NAME` in `.env`.
- The SQL guardrail is intentionally strict: only `SELECT` queries are allowed; anything else returns an error row in the JSON.
- The visualization logic in `app.py` is intentionally simple (first categorical vs first numeric column). You can replace it with richer, domain-specific charts using your preferred JS graph library, consuming the `response_json` structure.

<!-- ### Concept responses (Round 1)

**Q1) In your own words, explain what this problem is asking you to build and what kind of insights a leadership-level user should be able to obtain from your system.**

We are building a conversational analytics layer on top of the payments transaction schema, where a leadership user asks questions in plain language and we translate them into a pipeline of LLM-planned SQL tasks over DuckDB, aggregate the results into a unified JSON structure, and then turn that into narrative insights and visuals. Leaders should be able to explore trends (by time of day, day of week), segment behaviour (by age group, state, device, merchant category), and risk signals (failure and review rates, fraud flags) without writing any queries themselves.

**Q2) List three specific leadership or business questions that your system should be able to answer using the provided dataset schema.**

We have constructed a 100-row synthetic dataset that strictly follows the provided schema, and the current prototype pipeline (LLM planner → SQL over DuckDB → unified JSON → answer/visuals) is built and validated on top of this sample; an online version is available at [`https://techsas-analytics.streamlit.app/`](https://techsas-analytics.streamlit.app/). On top of this dataset, we want to reliably answer at least the following leadership questions:
- Which hours of the day have the highest P2P transaction count and highest total P2P value, and how do their failure rates compare?  
- For P2M transactions on weekends, which merchant categories contribute the most to failed volume (in INR) across states?  
- For senders aged 26–35, which device and network-type combinations show the highest share of high-value (₹5,000+) transactions flagged for review?

All of these can be mapped to concrete filters, group-bys, and aggregations over the existing columns in the schema.

**Q3) Choose one question from the previous response and explain how you would compute the answer using the dataset schema. Mention which columns would be used and what kind of comparison or aggregation would be performed. No code is required.**

For the weekend P2M failure-by-category question, we would filter `transaction_type = 'P2M'` and `is_weekend = 1`, then group by `merchant_category` (and optionally `sender_state`) and compute: total transaction count, count of `transaction_status = 'FAILED'`, and failed amount by summing `amount_inr` for failed rows. From there, we compute failure rates as `failed_count / total_count` and rank categories by failed amount and rate; these metrics flow through the same pipeline (LLM task plan → SQL → unified JSON) and are presented both as tables and category-wise charts.

**Q4) What assumptions or limitations will your system explicitly consider while answering questions from leadership?**

We assume the dataset is synthetic but internally consistent with the schema, that `fraud_flag` indicates “flagged for review” rather than confirmed fraud, and that `hour_of_day`, `day_of_week`, and `is_weekend` are correctly derived from `timestamp`. The system only answers questions that can be mapped to aggregations, filters, and comparisons over the provided columns; it does not infer causality, does not join in external sources, and treats historical slices as static snapshots rather than real-time telemetry. When a question exceeds what the schema supports, we surface that limitation explicitly in the natural-language layer.

**Q5) How will your system explain insights and answers to a non-technical, leadership-level user?**

We use the pipeline outputs (aggregated metrics and unified JSON) to drive short, structured narratives that highlight “what, how much, and so what” in clear terms, while still referencing the underlying segments (e.g., hour, day, device, age group, merchant category) and magnitudes. Each answer is paired with visuals backed by the same JSON—bar charts and tables that mirror the groupings used in SQL—so a leader can see both the explanation and the supporting numbers side by side. In the prototype we have intentionally kept the visuals minimal (single-chart plus table per primary task), but in the final solution we plan to invest more in multi-panel dashboards, side-by-side comparisons for key segments, and interaction patterns (filters, drill-down on categories or time ranges) that all remain grounded in the same unified JSON contract.

**Q6) What will your system deliberately not attempt to do as part of this problem?**

We deliberately stay within descriptive and comparative analytics over the provided schema and do not attempt user-level prediction, anomaly detection beyond simple rules based on existing columns, or any kind of automated decisioning. The system does not modify source data, does not ingest external payment streams, does not provide legal or compliance advice, and does not expose any personally identifiable information—its scope is to translate leadership questions into interpretable aggregations over the synthetic dataset and present those results cleanly.

**Q7) Briefly describe how your team plans to divide work and execute the solution if shortlisted.**

We plan to split execution into three tightly coupled tracks: (1) data and query layer, responsible for modelling the schema in DuckDB, validating distributions, and designing the key aggregation patterns; (2) orchestration and language layer, responsible for LangChain prompt design, LLM task planning, SQL generation, and unified JSON contracts; and (3) experience and delivery layer, responsible for the Streamlit interface, visualization wiring, deployment, and documentation. We iterate end-to-end on a small set of priority leadership questions, then expand coverage while keeping the same pipeline structure and contracts, while in parallel improving the presentation layer (richer visuals, clearer layouts) and optimising the orchestration so that request-to-response time stays low even as query complexity grows.
 -->
