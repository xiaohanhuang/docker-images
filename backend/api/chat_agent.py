"""Bedrock agent: orchestrates DeepSeek V3.2 with tool use for platform Q&A."""

import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an ML platform assistant. You help users understand their GPU cluster, \
training jobs, experiments, and infrastructure costs.

Rules:
1. Use the provided tools to fetch real data — never invent numbers.
2. Summarize the data you receive in a clear, concise response.
3. Do NOT discuss charts, visualization, dashboards, or the UI. \
   The system renders widgets automatically from tool data — your only job is to \
   call tools and explain what the data shows. Never say "I can't display charts" \
   or mention "frontend", "browser", or "display limitations".
4. For multi-step questions, chain tools: e.g., call query_mlflow_experiments first \
   to get experiment IDs, then call query_mlflow_runs for each experiment.
5. If the user asks about a specific job, use lookup_job_pods first, then query_prometheus.
6. For GPU metrics use DCGM_FI_DEV_GPU_UTIL. For CPU use \
   rate(node_cpu_seconds_total{mode!=\"idle\"}[5m]). For memory use node_memory_MemAvailable_bytes.
7. Always attempt the relevant tool call first. If a tool returns an error or empty data, \
   report what happened briefly — never refuse to try a tool.
8. Keep answers concise — 2-4 sentences for simple questions.
9. Never describe your own capabilities or limitations. Just act.
"""

MAX_TOOL_ROUNDS = 10

# ── Auto-widget generation from tool results ─────────────────────


def _auto_widget_from_prometheus(
    tool_input: dict, result: dict, is_instant: bool = False
) -> dict | None:
    """Convert Prometheus response into a widget spec."""
    data_section = result.get("data", {})
    results = data_section.get("result", [])
    if not results:
        return None

    promql = tool_input.get("promql", "metric")

    if is_instant:
        # Instant query → stat widget (one value) or table (multiple)
        if len(results) == 1:
            metric = results[0].get("metric", {})
            value = results[0].get("value", [0, "0"])
            label = metric.get("__name__", promql)
            return {
                "type": "stat",
                "title": label,
                "value": str(round(float(value[1]), 2)),
                "unit": "",
            }
        # Multiple series → table
        rows = []
        for r in results[:20]:
            metric = r.get("metric", {})
            val = r.get("value", [0, "0"])
            label = metric.get("__name__", "")
            instance = metric.get("instance", metric.get("pod", metric.get("node", "")))
            rounded_val = str(round(float(val[1]), 2))
            rows.append({"metric": label, "instance": instance, "value": rounded_val})
        return {
            "type": "table",
            "title": promql,
            "columns": ["metric", "instance", "value"],
            "rows": rows,
        }

    # Range query → line chart
    series_list = []
    all_data: dict[str, dict] = {}  # timestamp -> {time: ..., series_name: value}
    colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7c43", "#a05195", "#665191"]

    for idx, s in enumerate(results[:6]):
        metric = s.get("metric", {})
        label = metric.get("instance", metric.get("pod", metric.get("gpu", f"series_{idx}")))
        series_key = f"s{idx}"
        series_list.append(
            {
                "name": label[:30],
                "dataKey": series_key,
                "color": colors[idx % len(colors)],
            }
        )
        for ts, val in s.get("values", []):
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")
            if ts_str not in all_data:
                all_data[ts_str] = {"time": ts_str}
            try:
                all_data[ts_str][series_key] = round(float(val), 2)
            except (ValueError, TypeError):
                all_data[ts_str][series_key] = 0

    if not all_data:
        return None

    data_points = list(all_data.values())
    return {
        "type": "line",
        "title": promql[:60],
        "data": data_points,
        "xAxisKey": "time",
        "series": series_list,
        "live": True,
        "refreshQuery": promql,
    }


def _auto_widget_from_nodes(result: list) -> dict | None:
    """Convert K8s nodes list into a table widget."""
    if not result:
        return None
    rows = []
    for n in result:
        rows.append(
            {
                "name": n.get("name", ""),
                "status": n.get("status", ""),
                "type": n.get("instance_type", ""),
                "cpu": n.get("cpu_capacity", "0"),
                "gpu": n.get("gpu_capacity", "0"),
            }
        )
    return {
        "type": "table",
        "title": "Cluster Nodes",
        "columns": ["name", "status", "type", "cpu", "gpu"],
        "rows": rows,
    }


def _auto_widget_from_pods(result: list) -> dict | None:
    """Convert K8s pods list into a table widget."""
    if not result:
        return None
    rows = []
    for p in result[:50]:
        rows.append(
            {
                "name": p.get("name", ""),
                "namespace": p.get("namespace", ""),
                "status": p.get("status", ""),
                "node": p.get("node", ""),
                "gpu": p.get("gpu", "0"),
            }
        )
    return {
        "type": "table",
        "title": f"Pods ({len(result)} total)",
        "columns": ["name", "namespace", "status", "node", "gpu"],
        "rows": rows,
    }


def _auto_widget_from_cost(result: dict) -> dict | None:
    """Convert cost report into a bar chart or stat widget."""
    total = result.get("total_cost", 0)
    jobs = result.get("jobs", [])
    if not jobs:
        return {
            "type": "stat",
            "title": "Total Cost",
            "value": f"${total:,.2f}",
            "unit": "USD",
        }
    # Bar chart of cost per job
    data = []
    for j in jobs[:15]:
        data.append(
            {
                "job": j.get("workflow", j.get("job_id", ""))[:20],
                "cost": round(j.get("cost_usd", 0), 2),
            }
        )
    return {
        "type": "bar",
        "title": f"Cost Breakdown (${total:,.2f} total)",
        "data": data,
        "xAxisKey": "job",
        "series": [{"name": "Cost (USD)", "dataKey": "cost", "color": "#ff7c43"}],
    }


def _auto_widget_from_mlflow_experiments(result: list) -> dict | None:
    """Convert MLflow experiments into a table."""
    if not result:
        return None
    rows = []
    for exp in result[:30]:
        rows.append(
            {
                "id": exp.get("experiment_id", ""),
                "name": exp.get("name", ""),
                "stage": exp.get("lifecycle_stage", ""),
            }
        )
    return {
        "type": "table",
        "title": f"MLflow Experiments ({len(result)})",
        "columns": ["id", "name", "stage"],
        "rows": rows,
    }


def _auto_widget_from_mlflow_runs(result: list) -> dict | None:
    """Convert MLflow runs into a table."""
    if not result:
        return None
    rows = []
    for run in result[:30]:
        info = run.get("info", {})
        metrics = run.get("data", {}).get("metrics", [])
        metric_str = ", ".join(f"{m['key']}={m['value']:.3f}" for m in metrics[:3])
        rows.append(
            {
                "run_id": info.get("run_id", "")[:8],
                "status": info.get("status", ""),
                "metrics": metric_str,
            }
        )
    return {
        "type": "table",
        "title": f"MLflow Runs ({len(result)})",
        "columns": ["run_id", "status", "metrics"],
        "rows": rows,
    }


def _auto_widget_from_ray_jobs(result: list) -> dict | None:
    """Convert Ray jobs list into a table."""
    if not result:
        return None
    rows = []
    for j in result[:30]:
        rows.append(
            {
                "job_id": str(j.get("job_id", j.get("submission_id", "")))[:12],
                "status": j.get("status", ""),
                "runtime": j.get("runtime_env", {}).get("runtime_env", ""),
            }
        )
    return {
        "type": "table",
        "title": f"Ray Jobs ({len(result)})",
        "columns": ["job_id", "status", "runtime"],
        "rows": rows,
    }


def _unwrap_list(res: Any) -> Any:
    """Unwrap {"result": [...]} dicts back to lists for auto-widget builders."""
    if isinstance(res, dict) and "result" in res and isinstance(res["result"], list):
        return res["result"]
    return res


# Maps tool names → auto-widget builder functions
_WIDGET_BUILDERS: dict[str, Any] = {
    "query_prometheus": lambda inp, res: _auto_widget_from_prometheus(inp, res, False),
    "query_prometheus_instant": lambda inp, res: _auto_widget_from_prometheus(inp, res, True),
    "query_kubernetes_nodes": lambda inp, res: _auto_widget_from_nodes(_unwrap_list(res)),
    "query_kubernetes_pods": lambda inp, res: _auto_widget_from_pods(_unwrap_list(res)),
    "query_cost": lambda inp, res: _auto_widget_from_cost(res),
    "query_mlflow_experiments": lambda inp, res: _auto_widget_from_mlflow_experiments(
        _unwrap_list(res)
    ),
    "query_mlflow_runs": lambda inp, res: _auto_widget_from_mlflow_runs(_unwrap_list(res)),
    "query_ray_jobs": lambda inp, res: _auto_widget_from_ray_jobs(_unwrap_list(res)),
}


# ── Agent ────────────────────────────────────────────────────────


async def run_agent(
    user_message: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the Bedrock agent with tool use loop (non-streaming).

    Returns:
        Dict with 'text' and optional 'widget' keys.
    """
    collected_text = []
    widget = None

    async for event in run_agent_stream(user_message, history=history):
        if event["type"] == "text_delta":
            collected_text.append(event["content"])
        elif event["type"] == "widget":
            widget = event["content"]

    result: dict[str, Any] = {"text": "".join(collected_text)}
    if widget:
        result["widget"] = widget
    return result


async def run_agent_stream(
    user_message: str,
    history: list[dict[str, str]] | None = None,
):
    """Run the Bedrock agent with streaming token output.

    Yields dicts:
        {"type": "text_delta", "content": "..."} — streamed token chunks
        {"type": "widget", "content": {...}}     — auto-generated widget spec
        {"type": "done"}                         — stream complete
    """
    import boto3

    from backend.api.chat_tools import BEDROCK_TOOL_SPECS, TOOL_FUNCTIONS

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    messages = []
    if history:
        for msg in history:
            messages.append(
                {
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                }
            )

    messages.append({"role": "user", "content": [{"text": user_message}]})

    tool_config = {"tools": BEDROCK_TOOL_SPECS}

    # Track tool calls for auto-widget generation
    tool_call_log: list[dict] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = bedrock.converse_stream(
            modelId="deepseek.v3.2",
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
        )

        stop_reason = "end_turn"
        content_blocks: list[dict] = []
        current_text = ""
        current_tool_use: dict | None = None
        tool_input_buf = ""

        for event in response["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    current_tool_use = {
                        "toolUseId": start["toolUse"]["toolUseId"],
                        "name": start["toolUse"]["name"],
                    }
                    tool_input_buf = ""
                else:
                    current_text = ""

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    chunk = delta["text"]
                    current_text += chunk
                    yield {"type": "text_delta", "content": chunk}
                elif "toolUse" in delta:
                    tool_input_buf += delta["toolUse"].get("input", "")

            elif "contentBlockStop" in event:
                if current_tool_use is not None:
                    try:
                        parsed_input = json.loads(tool_input_buf) if tool_input_buf else {}
                    except json.JSONDecodeError:
                        parsed_input = {}
                    content_blocks.append(
                        {
                            "toolUse": {
                                **current_tool_use,
                                "input": parsed_input,
                            }
                        }
                    )
                    current_tool_use = None
                elif current_text:
                    content_blocks.append({"text": current_text})

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")

        messages.append({"role": "assistant", "content": content_blocks})

        if stop_reason != "tool_use":
            # Final answer — emit auto-generated widgets from tool call data.
            # Iterate in reverse to prefer the most recent (successful) call.
            for entry in reversed(tool_call_log):
                if "error" in entry["tool_result"]:
                    continue  # skip failed tool calls
                builder = _WIDGET_BUILDERS.get(entry["tool_name"])
                if builder:
                    try:
                        widget = builder(entry["tool_input"], entry["tool_result"])
                    except Exception:
                        logger.warning("Widget builder failed", exc_info=True)
                        continue
                    if widget:
                        yield {"type": "widget", "content": widget}
                        break  # one widget per response
            yield {"type": "done"}
            return

        # Handle tool calls
        tool_results = []
        for block in content_blocks:
            if "toolUse" not in block:
                continue

            tool_use = block["toolUse"]
            tool_name = tool_use["name"]
            tool_input = tool_use.get("input", {})
            tool_use_id = tool_use["toolUseId"]

            logger.info(f"Calling tool {tool_name} with input: {tool_input}")

            yield {
                "type": "text_delta",
                "content": f"\n> Querying {tool_name}…\n\n",
            }

            func = TOOL_FUNCTIONS.get(tool_name)
            if func is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    # Filter kwargs to only those the function accepts,
                    # so extra params from the LLM don't cause TypeErrors.
                    sig = inspect.signature(func)
                    valid_params = set(sig.parameters.keys())
                    filtered_input = (
                        {k: v for k, v in tool_input.items() if k in valid_params}
                        if valid_params
                        else tool_input
                    )
                    result = await func(**filtered_input)
                    if not isinstance(result, dict):
                        result = {"result": result}
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
                    result = {"error": str(e)}

            # Log for auto-widget generation
            tool_call_log.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_result": result,
                }
            )

            result_str = json.dumps(result, default=str)
            if len(result_str) > 20000:
                result = {"truncated": True, "preview": result_str[:20000]}

            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": result}],
                    }
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # Exhausted tool rounds — still try to emit widgets
    for entry in reversed(tool_call_log):
        if "error" in entry["tool_result"]:
            continue
        builder = _WIDGET_BUILDERS.get(entry["tool_name"])
        if builder:
            try:
                widget = builder(entry["tool_input"], entry["tool_result"])
            except Exception:
                continue
            if widget:
                yield {"type": "widget", "content": widget}
                break
    yield {"type": "done"}


def _parse_response(text: str) -> dict[str, Any]:
    """Parse the LLM response text into a ChatResponse-compatible dict.

    The LLM may return:
    - Plain text answer
    - JSON with {"text": "...", "widget": {...}}
    - JSON wrapped in ```json ... ``` blocks
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result: dict[str, Any] = {}
            if "text" in parsed:
                result["text"] = parsed["text"]
            if "widget" in parsed:
                result["widget"] = parsed["widget"]
            if not result:
                result["widget"] = parsed
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return {"text": text}
