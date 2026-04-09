"""Bedrock agent: orchestrates DeepSeek V3.2 with tool use for platform Q&A."""

import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

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

Autopsy Diagnostic (for "Analyze job {job_id}" requests):
- Act as a senior ML Engineer diagnosing the run.
- First call 'get_job_tasks' to get the per-task breakdown (node names, statuses, durations).
- Analyze EACH task individually — identify which tasks are slowest or failed.
- For each significant task, call 'get_task_source_code' with the task_name to review
  the actual training/processing code for optimization opportunities.
- Call 'get_job_metrics' for comprehensive NVIDIA DCGM metrics: GPU utilization,
  tensor core utilization, SM occupancy, memory utilization, framebuffer usage,
  GPU temperature, power usage, and PCIe throughput.
- Use DCGM metrics to diagnose: low tensor core util → not using mixed precision or
  Tensor Cores; low SM occupancy → kernel launch overhead or small batch; high FB used
  with low free → near OOM risk; low PCIe throughput → data loading bottleneck.
- Check logs (get_job_logs) for "Out of Memory", "OOM", or "CUDA error" keywords.
- In the source code, look for: batch_size, num_workers, gradient_accumulation,
  checkpoint frequency, DeepSpeed config, mixed precision, data loading bottlenecks.
- If GPU utilization is low, note that NVIDIA Nsight Systems profiling can pinpoint
  the bottleneck and is supported via `@gpu_task(nsight=True)`, the CLI `--nsight`
  flag, or setting `ML_PLAT_NSIGHT=1`. This injects the `nsys` binary via an init
  container. Recommend using `ml_platform_sdk.profiling.nsight_profile()` in the
  training code to capture a trace, then run `nsys stats <report>.nsys-rep` or
  download the `.nsys-rep` file for local analysis.
- Structure your response task by task, then provide overall recommendations.
- Use markdown formatting: headers (##), bold (**), bullet lists, and code blocks.
- Return concrete, actionable advice referencing specific code lines and parameters.
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


def _build_initial_messages(
    user_message: str, history: list[dict[str, str]] | None
) -> list[dict[str, Any]]:
    """Build the initial messages array from user input and history."""
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
    return messages


def _get_widget_from_log(tool_call_log: list[dict]) -> dict | None:
    """Attempt to generate a widget from the most recent successful tool call."""
    for entry in reversed(tool_call_log):
        if "error" in entry["tool_result"]:
            continue
        builder = _WIDGET_BUILDERS.get(entry["tool_name"])
        if builder:
            try:
                widget = builder(entry["tool_input"], entry["tool_result"])
                if widget:
                    return widget
            except Exception:
                logger.warning("Widget builder failed", exc_info=True)
                continue
    return None


async def _process_stream_events(
    stream: Any, content_blocks: list[dict]
) -> AsyncGenerator[dict | str, None]:
    """
    Iterate over the Bedrock stream, yield `text_delta` chunks,
    parse/append content blocks, and finally yield the `stop_reason` as a string.
    """
    stop_reason = "end_turn"
    current_text = ""
    current_tool_use: dict | None = None
    tool_input_buf = ""

    for event in stream:
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

    yield stop_reason


async def _execute_tools(
    content_blocks: list[dict], tool_call_log: list[dict]
) -> AsyncGenerator[dict | list[dict], None]:
    """Execute tools, yield text_delta for UI, and yield final tool results."""
    from backend.api.chat_tools import TOOL_FUNCTIONS

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
                # Filter kwargs to only those the function accepts.
                # If the function accepts **kwargs (VAR_KEYWORD), pass the full input
                # for forward compatibility; otherwise, drop any unexpected keys.
                sig = inspect.signature(func)
                params = sig.parameters
                accepts_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if accepts_var_kwargs:
                    filtered_input = tool_input
                else:
                    valid_params = {
                        name
                        for name, p in params.items()
                        if p.kind
                        in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    }
                    filtered_input = {k: v for k, v in tool_input.items() if k in valid_params}
                result = await func(**filtered_input)
                if not isinstance(result, dict):
                    result = {"result": result}
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                result = {"error": str(e)}

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
    yield tool_results


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
    """Run the Bedrock agent with streaming token output."""
    import boto3

    from backend.api.chat_tools import BEDROCK_TOOL_SPECS

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
    messages = _build_initial_messages(user_message, history)
    tool_config = {"tools": BEDROCK_TOOL_SPECS}
    tool_call_log: list[dict] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = bedrock.converse_stream(
            modelId="deepseek.v3.2",
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
        )

        content_blocks: list[dict] = []
        stop_reason = "end_turn"
        async for event in _process_stream_events(response["stream"], content_blocks):
            if isinstance(event, dict):
                yield event
            elif isinstance(event, str):
                stop_reason = event

        messages.append({"role": "assistant", "content": content_blocks})

        if stop_reason != "tool_use":
            widget = _get_widget_from_log(tool_call_log)
            if widget:
                yield {"type": "widget", "content": widget}
            yield {"type": "done"}
            return

        tool_results = []
        async for event in _execute_tools(content_blocks, tool_call_log):
            if isinstance(event, dict):
                yield event
            elif isinstance(event, list):
                tool_results = event

        messages.append({"role": "user", "content": tool_results})

    widget = _get_widget_from_log(tool_call_log)
    if widget:
        yield {"type": "widget", "content": widget}
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
