import json

import plotly.express as px
import pandas as pd
import streamlit as st

from insight_chain import run_insight_pipeline


st.set_page_config(
    page_title="InsightX Conversational Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("InsightX: Conversational Payments Analytics")
st.caption(
    "Ask natural language questions about the digital payments dataset. "
    "Behind the scenes, LangChain + Gemini plan tasks, run SQL over the dataset, "
    "and generate explainable insights."
)


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Pipeline**:

        1. Your question + schema → Gemini (task planner)
        2. Gemini outputs analysis tasks + SQL queries
        3. SQL runs on `dataset.csv` via DuckDB
        4. Unified JSON of results → Gemini (answer generator)
        5. You see:
           - Natural language answer
           - Visuals
           - Raw response JSON
        """
    )

user_question = st.text_area(
    "Ask a question about the transactions (e.g. *Which transaction type has the highest failure rate?*)",
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
        st.subheader("Natural language insight")
        st.write(answer)

        st.subheader("Visualizations (auto-generated from first task, if possible)")
        if not tasks:
            st.info("No tasks found in response JSON.")
        else:
            first_task = tasks[0]
            rows = first_task.get("rows", [])
            if not rows:
                st.info("First task has no rows to visualize.")
            else:
                df = pd.DataFrame(rows)

                # Try to infer a simple visualization:
                # - If there is a categorical column and a numeric column, build a bar chart.
                # - Otherwise, just show the table.
                categorical_cols = [
                    c for c in df.columns if df[c].dtype == "object" and c != "error"
                ]
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                if categorical_cols and numeric_cols:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "Could not infer a chart (no clear categorical + numeric columns). "
                        "Showing raw table instead."
                    )
                st.dataframe(df)

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

