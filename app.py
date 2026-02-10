import json

import plotly.express as px
import pandas as pd
import streamlit as st

from insight_chain import run_insight_pipeline


def _split_markdown_sections(text: str):
    """
    Lightweight parser to turn a markdown answer into (title, body) sections.

    It treats either:
    - Bold-only lines like **Title**, or
    - Markdown headings like # Title / ## Title / ### Title

    as section boundaries. The part before the first header is treated as an
    'Overview' section.
    """
    lines = text.splitlines()
    sections = []
    current_title = "Overview"
    current_lines = []

    for line in lines:
        stripped = line.strip()

        is_bold_header = (
            stripped.startswith("**")
            and stripped.endswith("**")
            and len(stripped) > 4
        )
        is_md_header = stripped.startswith("#")

        if is_bold_header or is_md_header:
            # Flush previous section
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))

            if is_bold_header:
                # Remove surrounding ** markers
                header_text = stripped.strip("* ").strip()
            else:
                # Remove leading #'s and surrounding whitespace
                header_text = stripped.lstrip("#").strip()

            current_title = header_text or current_title
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections


def _pick_metric_column(numeric_cols):
    """
    Choose a "best" metric column from a list of numeric column names.
    Preference is given to rate/percentage-style metrics, then counts, then
    generic numeric fields.
    """
    preferred_order = [
        "success_rate_percentage",
        "failed_rate_percentage",
        "flagged_percentage",
        "flagged_rate",
        "failure_rate",
        "failed_transactions",
        "successful_transactions",
        "total_transactions",
        "amount_inr",
    ]
    for name in preferred_order:
        if name in numeric_cols:
            return name
    return numeric_cols[0] if numeric_cols else None


def _render_task_summary(df: pd.DataFrame, task_name: str):
    """
    Render small KPI-style highlights for a task:
    - total of the main metric
    - top segment (if categorical present)
    - number of groups / rows
    """
    if df.empty:
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return

    y_col = _pick_metric_column(numeric_cols)
    if not y_col:
        return

    total_val = df[y_col].sum()
    top_row = df.sort_values(y_col, ascending=False).iloc[0]

    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c != "error"]
    top_label = None
    if categorical_cols:
        top_label = f"{categorical_cols[0]} = {top_row[categorical_cols[0]]}"

    col1, col2, col3 = st.columns(3)

    def _fmt(v):
        if pd.isna(v):
            return "-"
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{float(v):,.2f}"

    with col1:
        st.metric(label=f"Total {y_col}", value=_fmt(total_val))

    with col2:
        if top_label is not None:
            st.metric(label=f"Top segment by {y_col}", value=_fmt(top_row[y_col]), delta=top_label)
        else:
            st.metric(label=f"Max {y_col}", value=_fmt(top_row[y_col]))

    with col3:
        st.metric(label="Groups / rows", value=str(len(df)))


def _render_task_chart(df: pd.DataFrame, task_name: str):
    """
    Heuristic visualisation helper:
    - If we have a time-like axis (hour_of_day or day_of_week), use a line/bar chart over that.
    - If we have two categorical axes and a metric, use a heatmap.
    - Otherwise, fall back to a categorical vs numeric bar chart.
    """
    if df.empty:
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c != "error"]

    if not numeric_cols:
        st.info("No numeric columns available for charting; showing table only.")
        return

    y_col = _pick_metric_column(numeric_cols)

    # Special handling for hour_of_day
    if "hour_of_day" in df.columns and y_col:
        fig = px.line(
            df.sort_values("hour_of_day"),
            x="hour_of_day",
            y=y_col,
            markers=True,
            title=f"{task_name} – {y_col} by hour of day",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Special handling for day_of_week ordering
    if "day_of_week" in df.columns and y_col:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if set(df["day_of_week"]).issubset(set(order)):
            df = df.copy()
            df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=order, ordered=True)
        fig = px.bar(
            df.sort_values("day_of_week"),
            x="day_of_week",
            y=y_col,
            title=f"{task_name} – {y_col} by day of week",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # If we have two categorical columns and one metric, show a heatmap
    if len(categorical_cols) >= 2 and y_col:
        x_col, y_cat = categorical_cols[0], categorical_cols[1]
        fig = px.density_heatmap(
            df,
            x=x_col,
            y=y_cat,
            z=y_col,
            color_continuous_scale="Blues",
            title=f"{task_name} – {y_col} heatmap by {x_col} and {y_cat}",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Generic categorical vs numeric bar chart
    if categorical_cols and y_col:
        x_col = categorical_cols[0]
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=f"{task_name} – {x_col} vs {y_col}",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Fallback: no suitable chart, table only
    st.info("Could not infer a meaningful chart for this task; showing table only.")

st.set_page_config(
    page_title="InsightX Conversational Analytics Agent",
    page_icon="🔍",
    layout="wide",
)

st.title("InsightX: Conversational Payments Analytics")
st.caption(
    "Ask natural language questions on top of the digital payments dataset. "
    "Behind the scenes, a LangChain-driven orchestration layer decomposes your prompt into analysis tasks, "
    "generates SQL over a DuckDB-backed `transactions` table, aggregates the results into a unified JSON contract, "
    "and then renders explainable insights (e.g. Ask anything like: *Which hours of the day have the highest P2P transaction count and highest total P2P value, and how do their failure rates compare?*)"
)


with st.sidebar:
    st.header("Pipeline:")
    st.markdown(
        """
        1. User prompt + `schema` → **LangChain-powered LLM Task Planner**  
           A LangChain `PromptTemplate` + `ChatGoogleGenerativeAI` chain conditions the LLM on the natural-language question and the full dataset schema, and materializes an explicit *task graph* of analytical intents (each with `task_name`, `task_description`, and a targeted objective).

        2. Task graph → **LLM-to-SQL semantic translation**  
           A LangChain JSON-output chain converts each task into a parameterized, read-only SQL query over the DuckDB-backed `transactions` relation (sourced from [`dataset.csv`](https://github.com/techSaswata/Transactions_Analytics-Demo/blob/main/dataset.csv)), encoding all filters, group-bys, windowing, and aggregation logic directly in SQL.

        3. DuckDB execution → **LangChain-normalized unified response JSON**  
           The application executes the generated SQL via DuckDB, then normalizes all result sets into a single, strongly-typed unified JSON envelope (one node per task with metadata + row-level payload). This structure is the canonical data contract between the query engine, LangChain flows, and the visualization layer.

        4. Unified JSON + original prompt → **LLM Insight Generation Chain**  
           A second LangChain conversation (LLM + system prompt) consumes both the original leadership question and the unified JSON to synthesize an explainable, narrative-style answer, strictly grounded in the computed metrics and aligned with the explainability requirements.

        5. Frontend / UX layer  
           The Streamlit UI exposes three synchronized facets of the same LangChain pipeline:
           - **Natural language output** – the final LLM-generated insight layer  
           - **Visuals** – Plotly-based charts/tables driven by the unified JSON and ready to be swapped for any graphing library  
           - **Response JSON** – the low-level machine-readable contract suitable for downstream services, dashboards, or further LangChain tooling
        """
    )

user_question = st.text_area(
    "",
    height=100,
)

if st.button("Run analysis") and user_question.strip():
    with st.spinner("Thinking with Gemini, planning tasks, and running SQL..."):
        try:
            result = run_insight_pipeline(user_question.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {e}")
            st.stop()

    answer = result["answer"]
    response_json = result["response_json"]
    tasks = response_json.get("tasks", [])

    # Layout: 2 columns – answer & visuals on left, JSON on right
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Leadership insight console")
        # Render markdown as-is, similar to GitHub's markdown rendering
        if not isinstance(answer, str):
            answer = str(answer)
        st.markdown(answer)

        st.subheader("Visualizations")
        if not tasks:
            st.info("No tasks found in response JSON.")
        else:
            tabs = st.tabs([f"{idx+1}. {t.get('task_name', 'Task')}" for idx, t in enumerate(tasks)])
            for idx, (tab, t) in enumerate(zip(tabs, tasks), start=1):
                with tab:
                    task_name = t.get("task_name") or f"Task {idx}"
                    rows = t.get("rows", [])
                    if not rows:
                        st.caption("No result rows returned for this task.")
                        continue

                    df = pd.DataFrame(rows)
                    _render_task_summary(df, task_name)
                    _render_task_chart(df, task_name)
                    st.dataframe(df, use_container_width=True)

    with col_right:
        st.subheader("Analysis breakdown")

        if not tasks:
            st.info("No tasks found to display.")
        else:
            # High-level task plan view (what the planner LLM decided to do)
            with st.expander("Task plan (LLM decomposition)", expanded=False):
                for idx, t in enumerate(tasks, start=1):
                    task_name = t.get("task_name") or f"Task {idx}"
                    task_desc = t.get("task_description") or "_No description provided_"
                    sql_query = t.get("sql_query") or "-- No SQL generated --"

                    st.markdown(f"**{idx}. {task_name}**")
                    st.markdown(task_desc)
                    with st.expander("View SQL query", expanded=False):
                        st.code(sql_query, language="sql")
                    st.markdown("---")

            # Execution results view – parsed and tabular, suitable for non-technical users
            with st.expander("Execution results (DB output)", expanded=False):
                for idx, t in enumerate(tasks, start=1):
                    task_name = t.get("task_name") or f"Task {idx}"
                    rows = t.get("rows", [])

                    st.markdown(f"**Task {idx}: {task_name}**")

                    if not rows:
                        st.caption("No result rows returned for this task.")
                        st.markdown("---")
                        continue

                    df_task = pd.DataFrame(rows)

                    # If there is an error column, show it prominently instead of the full table
                    if "error" in df_task.columns and df_task["error"].notna().any():
                        st.error(df_task["error"].iloc[0])
                        if "sql_received" in df_task.columns:
                            with st.expander("SQL received (from planner)", expanded=False):
                                st.code(str(df_task["sql_received"].iloc[0]), language="sql")
                    else:
                        st.dataframe(df_task, use_container_width=True, height=250)

                    st.markdown("---")

            # Optional: raw JSON for debugging / dev usage
            with st.expander("Raw JSON (developer view)", expanded=False):
                st.json(response_json)

