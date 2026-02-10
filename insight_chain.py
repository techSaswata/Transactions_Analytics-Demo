import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# Default to Gemini 3 Pro preview; can be overridden via GEMINI_MODEL_NAME in .env
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-pro-preview")


@dataclass
class TaskPlanItem:
    task_name: str
    task_description: str
    sql_query: str


def _make_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment (.env).")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )


def load_transactions_df(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """
    Loads the synthetic transaction dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV '{csv_path}' not found. Make sure dataset.csv exists in the project root."
        )
    return pd.read_csv(csv_path)


def _schema_text() -> str:
    """
    Loads the human-readable schema description from schema.txt
    so it can be injected into prompts.
    """
    path = os.path.join(os.path.dirname(__file__), "schema.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def plan_tasks(user_question: str) -> List[TaskPlanItem]:
    """
    First LLM call:
    - Takes the natural language question + schema description
    - Produces a list of analysis tasks with SQL queries over the transactions table.
    """
    llm = _make_llm(temperature=0.0)

    schema = _schema_text()

    prompt = PromptTemplate(
        input_variables=["question", "schema"],
        template=(
            "You are a senior analytics engineer for a digital payments product.\n"
            "You are given:\n"
            "1) A human-readable description of a transaction dataset schema.\n"
            "2) A leadership-level natural language analytics question.\n\n"
            "Dataset notes:\n"
            "{schema}\n\n"
            "The data is available as a DuckDB table named transactions, with columns\n"
            "matching the dataset description (transaction_id, timestamp, transaction_type, "
            "merchant_category, amount_inr, transaction_status, sender_age_group, "
            "receiver_age_group, sender_state, sender_bank, receiver_bank, device_type, "
            "network_type, fraud_flag, hour_of_day, day_of_week, is_weekend).\n\n"
            "Task:\n"
            "Break the question into a SMALL list of 1-4 atomic analysis tasks.\n"
            "Each task should:\n"
            "- Focus on a single clear analytical goal (e.g., compare failure rates by device_type).\n"
            "- Include an expressive but SAFE SQL query over the transactions table.\n\n"
            "Very important requirements:\n"
            "- Only SELECT queries are allowed.\n"
            "- Do NOT use DDL or DML (no CREATE, INSERT, UPDATE, DELETE, DROP, etc.).\n"
            "- Use column names exactly as described.\n"
            "- If you filter on categorical columns, only use valid values from the schema.\n"
            "- Make sure queries are syntactically valid DuckDB SQL.\n"
            "- If the question is already simple, you may return just 1 task.\n\n"
            "Return STRICTLY valid JSON with the following structure:\n"
            "{{\n"
            '  "tasks": [\n'
            "    {{\n"
            '      "task_name": "short title",\n'
            '      "task_description": "what this task will compute and why",\n'
            '      "sql_query": "SELECT ... FROM transactions ..."\n'
            "    }}\n"
            "  ]\n"
            "}}\n\n"
            "User question:\n"
            "{question}\n"
        ),
    )

    chain = prompt | llm | JsonOutputParser()

    raw = chain.invoke({"question": user_question, "schema": schema})
    tasks: List[TaskPlanItem] = []
    for t in raw.get("tasks", []):
        tasks.append(
            TaskPlanItem(
                task_name=t.get("task_name", "Unnamed Task"),
                task_description=t.get("task_description", ""),
                sql_query=t.get("sql_query", ""),
            )
        )
    return tasks


def execute_tasks(tasks: List[TaskPlanItem], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes each planned SQL query against the DuckDB in-memory DB.

    Returns a unified JSON-serializable structure:
    {
      "tasks": [
        {
          "task_name": ...,
          "task_description": ...,
          "sql_query": ...,
          "rows": [ {col: value, ...}, ... ]
        },
        ...
      ]
    }
    """
    con = duckdb.connect(database=":memory:")
    con.register("transactions", df)

    unified: Dict[str, Any] = {"tasks": []}

    for t in tasks:
        sql = t.sql_query.strip()

        # Simple guardrail: only allow SELECT queries.
        if not sql.lower().lstrip().startswith("select"):
            rows: List[Dict[str, Any]] = [
                {
                    "error": "Only SELECT queries are allowed.",
                    "sql_received": sql,
                }
            ]
        else:
            try:
                result_df = con.execute(sql).df()

                # Ensure all values are JSON-serializable (e.g. convert timestamps to strings)
                datetime_cols = result_df.select_dtypes(
                    include=["datetime64[ns]", "datetime64[ns, tz]"]
                ).columns
                for col in datetime_cols:
                    # Use a consistent, human-readable format
                    result_df[col] = result_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

                rows = result_df.to_dict(orient="records")
            except Exception as e:  # noqa: BLE001
                rows = [
                    {
                        "error": f"Query execution failed: {e}",
                        "sql_received": sql,
                    }
                ]

        unified["tasks"].append(
            {
                "task_name": t.task_name,
                "task_description": t.task_description,
                "sql_query": t.sql_query,
                "rows": rows,
            }
        )

    return unified


def generate_answer(user_question: str, unified_json: Dict[str, Any]) -> str:
    """
    Second LLM call:
    - Takes the original question + unified JSON with task-level results
    - Produces a leadership-ready, explainable answer.
    """
    llm = _make_llm(temperature=0.2)

    system_content = (
        "You are an AI assistant for business leaders at a digital payments company.\n"
        "You are given:\n"
        "1) A leadership-level natural language question.\n"
        "2) A JSON structure containing analysis tasks and their SQL results over a\n"
        "   synthetic digital payments transaction dataset.\n\n"
        "Your job is to:\n"
        "- Directly answer the question.\n"
        "- Use the provided numbers and trends from the JSON; do NOT invent data.\n"
        "- Provide clear, explainable reasoning behind conclusions.\n"
        "- Highlight key statistics and trends.\n"
        "- Where appropriate, add 1-3 concise recommendations.\n\n"
        "Do NOT output JSON. Respond in natural language paragraphs, suitable for\n"
        "a senior product/operations/marketing/risk leader.\n"
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=(
                f"User question:\n{user_question}\n\n"
                "Analysis JSON (from previous step):\n"
                f"{json.dumps(unified_json, indent=2)}\n\n"
                "Now provide the final, well-structured answer."
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def run_insight_pipeline(user_question: str) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      user prompt -> (schema-aware task planner LLM) -> SQL per task
      -> DuckDB execution over dataset.csv -> unified JSON
      -> answer-generation LLM -> natural language answer

    Returns:
    {
      "answer": "<natural language answer>",
      "response_json": { ... unified JSON ... }
    }
    """
    df = load_transactions_df()
    tasks = plan_tasks(user_question)
    unified_json = execute_tasks(tasks, df)
    answer = generate_answer(user_question, unified_json)

    return {
        "answer": answer,
        "response_json": unified_json,
    }

