import asyncio
import traceback
from dotenv import load_dotenv
import chainlit as cl

from helpers.followup_profile import followup_profile
from helpers.ai_brief_from_metrics import ai_brief_from_metrics
from helpers.ai_followups import suggest_followups
from helpers.ai_holiday_impact import ai_explain_holiday_impact
from helpers.ai_qa import ai_answer_question, should_use_llm_for_question
from ingest import load_table, load_table_from_text
from profiling import profile_df
from semantic import suggest_mapping, apply_mapping_override, validate_temporal_mapping
from metrics import compute_all_metrics
from qa import answer_question

from data_sources.enrichment_engine import connect_datasource
from helpers.ensure_derived import ensure_derived
from helpers.enrichment_recommender import recommend_enrichment

load_dotenv()


CORE_MAPPING_KEYS = [
    "id",
    "status",
    "amount",
    "currency",
    "created_at",
    "paid_at",
    "expected_arrival_at",
    "arrival_at",
]


def _mapping_summary_markdown(mapping: dict | None, max_items: int = 20) -> str:
    mapping = mapping or {}

    ordered_keys = [k for k in CORE_MAPPING_KEYS if k in mapping]
    ordered_keys.extend(sorted([k for k in mapping.keys() if k not in CORE_MAPPING_KEYS]))

    if not ordered_keys:
        return "- _(no mapping inferred)_"

    visible_keys = ordered_keys[:max_items]
    lines = []
    for key in visible_keys:
        value = mapping.get(key)
        value_text = str(value) if value else "_(not mapped)_"
        lines.append(f"- {key}: `{value_text}`")

    remaining = len(ordered_keys) - len(visible_keys)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more mapped fields")

    return "\n".join(lines)


def _to_preview_markdown(df, max_rows: int = 8, max_cols: int = 10) -> str:
    preview = df.head(max_rows)
    cols = list(preview.columns)[:max_cols]
    preview = preview[cols] if cols else preview

    if preview.empty:
        return "| No data available |\n|---|"

    headers = [str(col) for col in preview.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_rows = []
    for _, row in preview.iterrows():
        values = []
        for value in row.tolist():
            if value is None:
                values.append("")
            else:
                text = str(value).replace("\n", " ").replace("|", "\\|")
                values.append(text)
        body_rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header_row, separator_row, *body_rows])


def _is_delay_explanation_question(question: str) -> bool:
    q = (question or "").lower()
    has_delay_topic = any(token in q for token in ["delay", "delays", "late", "lateness", "settlement", "arrival time", "transfer time"])
    has_explain_intent = any(token in q for token in ["why", "explain", "reason", "driver", "drivers", "cause", "causes"])
    return has_delay_topic and has_explain_intent


def _fallback_openholidays_suggestion(df, mapping: dict) -> dict | None:
    if df is None:
        return None

    if "country" not in df.columns:
        return None

    exp_col = (mapping or {}).get("expected_arrival_at")
    if not exp_col or exp_col not in df.columns:
        return None

    return {
        "source_id": "openholidays",
        "reason": "Delay explanations may depend on holiday calendar effects that are not in the current dataset.",
        "confidence": 0.78,
    }


async def _suggest_enrichment_if_relevant(df, mapping: dict, question: str, response: str) -> None:
    if not cl.user_session.get("mapping_ready", False):
        return

    connected_sources = set(cl.user_session.get("connected_datasources") or [])

    suggestion = await recommend_enrichment(df, mapping, question or "", response or "")
    if not suggestion and _is_delay_explanation_question(question or ""):
        suggestion = _fallback_openholidays_suggestion(df, mapping)

    if suggestion and suggestion.get("source_id") in connected_sources:
        suggestion = None

    if suggestion:
        await cl.Message(
            content=(
                "I found a potentially useful external enrichment.\n\n"
                f"- Source: `{suggestion['source_id']}`\n"
                f"- Why: {suggestion['reason']}\n"
                f"- Confidence: {suggestion['confidence']:.2f}"
            ),
            actions=[
                cl.Action(
                    name="connect_datasource",
                    payload={"source_id": suggestion["source_id"]},
                    label=f"Connect {suggestion['source_id']}",
                    icon="plug",
                )
            ],
        ).send()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("df", None)
    cl.user_session.set("df_enriched", None)
    cl.user_session.set("mapping", {})
    cl.user_session.set("holidays_connected", False)
    cl.user_session.set("connected_datasources", [])
    cl.user_session.set("mapping_ready", False)
    cl.user_session.set("pending_mapping_target", None)
    await cl.Message(
        content=(
            "Upload a CSV/XLSX to begin or paste the data here. \n"
            "You can also get a sample dataset at: \n"
            "https://github.com/alexanmtz/horizon-analytics/tree/main/sample_data"
        )
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    text = (msg.content or "").strip()
    loaded_df = None
    loaded_mapping = None

    # 1) File upload
    if msg.elements:
        file_el = next((e for e in msg.elements if hasattr(e, "path")), None)
        if not file_el:
            await cl.Message(content="No file detected or data pasted. Upload CSV/XLSX or paste the data here.").send()
            return

        async with cl.Step(name="Process upload") as step:
            step.output = "Reading uploaded file..."
            df = await asyncio.to_thread(load_table, file_el.path)
            cl.user_session.set("df", df)
            cl.user_session.set("df_enriched", None)
            cl.user_session.set("holidays_connected", False)
            cl.user_session.set("connected_datasources", [])
            cl.user_session.set("mapping_ready", False)
            cl.user_session.set("pending_mapping_target", None)

            step.output = "Inferring semantic column mapping..."
            mapping = await asyncio.to_thread(suggest_mapping, df)
            cl.user_session.set("mapping", mapping)

            step.output = f"Loaded {len(df)} rows and prepared mapping."
            loaded_df = df
            loaded_mapping = mapping


    # 2) Pasted tabular text
    has_dataset = cl.user_session.get("df") is not None or cl.user_session.get("df_enriched") is not None
    if text and not has_dataset and _looks_like_tabular_text(text):
        async with cl.Step(name="Process pasted data") as step:
            step.output = "Reading pasted data..."
            try:
                df = await asyncio.to_thread(load_table_from_text, text)
            except Exception as exc:
                await cl.Message(
                    content=(
                        "I couldn't parse the pasted data. "
                        "Please paste CSV/TSV content including a header row.\n\n"
                        f"Details: {exc}"
                    )
                ).send()
                return

            cl.user_session.set("df", df)
            cl.user_session.set("df_enriched", None)
            cl.user_session.set("holidays_connected", False)
            cl.user_session.set("connected_datasources", [])
            cl.user_session.set("mapping_ready", False)
            cl.user_session.set("pending_mapping_target", None)

            step.output = "Inferring semantic column mapping..."
            mapping = await asyncio.to_thread(suggest_mapping, df)
            cl.user_session.set("mapping", mapping)

            step.output = f"Loaded {len(df)} rows from pasted data and prepared mapping."
        loaded_df = df
        loaded_mapping = mapping

    if loaded_df is not None:
        sanitized_mapping, mapping_warnings = validate_temporal_mapping(loaded_df, loaded_mapping or {})
        cl.user_session.set("mapping", sanitized_mapping)
        data_preview_md = _to_preview_markdown(loaded_df, 8, 10)

        warning_text = ""
        if mapping_warnings:
            warning_text = (
                "\n\nTemporal mapping guard:\n"
                + "\n".join([f"- {w}" for w in mapping_warnings])
            )

        await cl.Message(
            content=(
                f"Loaded {len(loaded_df)} rows.\n\n"
                "### Data Preview\n\n"
                f"{data_preview_md}\n\n"
                "Ask questions about this data, generate insights, or suggest enrichment."
                f"{warning_text}"
            ),
            actions=[
                cl.Action(name="generate_insights", payload={"action": "generate_insights"}, label="Generate Insights", icon="sparkles"),
                cl.Action(name="suggest_enrichment", payload={"reason": "post_upload"}, label="Suggest Enrichment", icon="plug"),
                cl.Action(name="show_profile", payload={"action": "profile"}, label="View Profile", icon="bar-chart-3"),
            ],
        ).send()
        return

    # 3) Commands
    if text.lower().startswith("map "):
        df = cl.user_session.get("df")
        if df is None:
            await cl.Message(content="Upload a dataset first.").send()
            return

        mapping = cl.user_session.get("mapping") or {}
        new_mapping = apply_mapping_override(mapping, text)
        cl.user_session.set("mapping", new_mapping)

        await cl.Message(
            content=(
                "Mapping updated.\n"
                f"{_mapping_summary_markdown(new_mapping)}\n\n"
                "Now click Confirm."
            ),
            actions=[
                cl.Action(name="confirm_mapping", payload={
                              "action": "confirm"}, label="Confirm", icon="check"),
            ],
        ).send()
        return

    # 4) Normal Q&A
    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")
    if df is None:
        await cl.Message(content="Upload or paste a dataset first.").send()
        return

    mapping = cl.user_session.get("mapping") or {}
    sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, mapping)
    if sanitized_mapping != mapping:
        cl.user_session.set("mapping", sanitized_mapping)
    mapping = sanitized_mapping
    used_ai_deep_analysis = False

    async with cl.Step(name="Answer question") as step:
        step.output = "Running analytics Q&A..."
        user_question = msg.content or ""
        response = await asyncio.to_thread(answer_question, df, mapping, user_question)

        if should_use_llm_for_question(user_question, response or ""):
            step.output = "Generating deeper AI answer..."
            try:
                response = await ai_answer_question(df, mapping, user_question, response or "")
                used_ai_deep_analysis = True
            except Exception:
                print("AI QA fallback error:\n", traceback.format_exc())

        step.output = "Generating suggested follow-up questions..."
        profile = await asyncio.to_thread(followup_profile, df, mapping)
        followups = await suggest_followups(msg.content or "", response, profile)

        step.output = "Q&A complete."

    # Send the main answer
    final_response = response or "DEBUG: QA returned empty"
    if used_ai_deep_analysis:
        final_response = f"_AI deep analysis_\n\n{final_response}"
    if mapping_warnings:
        mapping_warning_text = "\n\n_Temporal mapping guard:_\n" + "\n".join([f"- {w}" for w in mapping_warnings])
        final_response = f"{final_response}{mapping_warning_text}"
    await cl.Message(content=final_response).send()

    await _suggest_enrichment_if_relevant(df, mapping, msg.content or "", response or "")

    # Show clickable followups
    await cl.Message(
        content="### Suggested next questions",
        actions=[
            cl.Action(name="ask_followup", payload={"q": q}, label=q)
            for q in followups
        ],
    ).send()

    return


def _looks_like_tabular_text(text: str) -> bool:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    delimiters = [",", "\t", ";", "|"]
    for delim in delimiters:
        header_count = lines[0].count(delim)
        second_count = lines[1].count(delim)
        if header_count > 0 and second_count == header_count:
            return True

    return False


@cl.action_callback("show_profile")
async def show_profile(action: cl.Action):
    df = cl.user_session.get("df")
    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return
    prof = profile_df(df)
    await cl.Message(content=f"Rows: {prof['rows']}, Cols: {prof['cols']}").send()


@cl.action_callback("apply_mapping_override")
async def apply_mapping_override_action(action: cl.Action):
    df = cl.user_session.get("df")
    if df is None:
        await cl.Message(content="Upload a dataset first.").send()
        return

    command = ((action.payload or {}).get("command") or "").strip()
    if not command.lower().startswith("map "):
        await cl.Message(content="Invalid override command.").send()
        return

    mapping = cl.user_session.get("mapping") or {}
    new_mapping = apply_mapping_override(mapping, command)
    sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, new_mapping)
    cl.user_session.set("mapping", sanitized_mapping)

    warning_text = ""
    if mapping_warnings:
        warning_text = "\n\nTemporal mapping guard:\n" + "\n".join([f"- {w}" for w in mapping_warnings])

    pending_mapping_target = cl.user_session.get("pending_mapping_target")
    if pending_mapping_target == "insights":
        confirm_action_name = "confirm_mapping_for_insights"
    elif pending_mapping_target == "enrichment":
        confirm_action_name = "confirm_mapping_for_enrichment"
    else:
        confirm_action_name = "confirm_mapping"

    await cl.Message(
        content=(
            "Mapping updated.\n"
            f"{_mapping_summary_markdown(sanitized_mapping)}\n\n"
            "Now click Confirm."
            f"{warning_text}"
        ),
        actions=[
            cl.Action(name=confirm_action_name, payload={"action": "confirm"}, label="Confirm", icon="check"),
        ],
    ).send()


@cl.action_callback("confirm_mapping")
async def confirm_mapping(action: cl.Action):
    await _prepare_data_preview(
        step_name="Confirm mapping",
        completion_step_output="Mapping confirmed and data preview ready.",
        summary_prefix="Mapping confirmed.",
    )


@cl.action_callback("skip_mapping")
async def skip_mapping(action: cl.Action):
    await cl.Message(content="Keeping the data as is (no mapping override applied).", author="Assistant").send()

    await _prepare_data_preview(
        step_name="Skip mapping override",
        completion_step_output="Kept mapping as-is and prepared data preview.",
        summary_prefix="Kept the data as is.",
    )


async def _prepare_data_preview(
    step_name: str,
    completion_step_output: str,
    summary_prefix: str,
):
    df_enriched = cl.user_session.get("df_enriched")
    df = df_enriched if df_enriched is not None else cl.user_session.get("df")
    mapping = cl.user_session.get("mapping") or {}

    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return

    sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, mapping)
    if sanitized_mapping != mapping:
        cl.user_session.set("mapping", sanitized_mapping)
    mapping = sanitized_mapping

    warning_text = ""
    if mapping_warnings:
        warning_text = (
            "\n\n### Temporal mapping guard\n"
            + "\n".join([f"- {w}" for w in mapping_warnings])
        )

    async with cl.Step(name=step_name) as step:
        step.output = "Preparing derived columns..."
        derived = await asyncio.to_thread(ensure_derived, df, mapping)
        cl.user_session.set("df_enriched", derived)

        step.output = "Preparing data preview..."
        data_preview_md = await asyncio.to_thread(_to_preview_markdown, derived, 8, 10)

        step.output = completion_step_output

    preview_content = (
        f"{summary_prefix}\n\n"
        "### Data Preview\n\n"
        f"{data_preview_md}\n\n"
    )
    if warning_text:
        preview_content += f"{warning_text}\n\n"
    preview_content += "Ask a question to explore drivers, or click below to generate the AI insight brief."

    await cl.Message(
        content=preview_content,
        actions=[
            cl.Action(name="generate_insights", payload={"action": "generate_insights"}, label="Generate Insights", icon="sparkles"),
            cl.Action(name="suggest_enrichment", payload={"reason": "data_preview"}, label="Suggest Enrichment", icon="plug"),
        ],
    ).send()


@cl.action_callback("generate_insights")
async def generate_insights(action: cl.Action):
    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")
    mapping = cl.user_session.get("mapping") or {}
    metrics_text = cl.user_session.get("latest_metrics_text")

    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return

    if not cl.user_session.get("mapping_ready", False):
        sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, mapping)
        cl.user_session.set("mapping", sanitized_mapping)
        cl.user_session.set("pending_mapping_target", "insights")

        warning_text = ""
        if mapping_warnings:
            warning_text = "\n\nTemporal mapping guard:\n" + "\n".join([f"- {w}" for w in mapping_warnings])

        await cl.Message(
            content=(
                "Before generating insights, confirm the temporal mapping.\n\n"
                "Suggested mapping:\n"
                f"{_mapping_summary_markdown(sanitized_mapping)}\n\n"
                "You can override mapping if needed, then confirm."
                f"{warning_text}"
            ),
            actions=[
                cl.Action(
                    name="apply_mapping_override",
                    payload={
                        "command": (
                            "map "
                            f"paid_at={(sanitized_mapping or {}).get('paid_at') or ''} "
                            f"arrival_at={(sanitized_mapping or {}).get('arrival_at') or ''} "
                            f"expected_arrival_at={(sanitized_mapping or {}).get('expected_arrival_at') or ''}"
                        ).strip()
                    },
                    label="Override Mapping",
                    icon="wrench",
                ),
                cl.Action(name="skip_mapping_for_insights", payload={"action": "skip"}, label="Skip", icon="skip-forward"),
            ],
        ).send()
        return

    if not metrics_text:
        metrics_text = await asyncio.to_thread(compute_all_metrics, df, mapping)
        cl.user_session.set("latest_metrics_text", metrics_text)

    brief = ""
    ai_task = None
    async with cl.Step(name="Generate insight brief") as step:
        step.output = "Generating AI insight brief..."
        try:
            ai_task = asyncio.create_task(
                ai_brief_from_metrics(df, mapping, metrics_text, timeout_seconds=15.0)
            )
            done, _ = await asyncio.wait({ai_task}, timeout=20)
            if done:
                brief = ai_task.result()
            else:
                brief = "_(AI brief timed out.)_"
                ai_task.cancel()
        except Exception:
            brief = "_(AI brief failed.)_"
            if ai_task and not ai_task.done():
                ai_task.cancel()
            print("AI brief error:\n", traceback.format_exc())
        step.output = "Insight brief ready."

    await cl.Message(
        content=(
            "### Initial Insight Brief\n\n"
            f"{brief}\n\n"
            "Ask a question to explore this data further."
        ),
        actions=[
            cl.Action(name="suggest_enrichment", payload={"reason": "initial_insights"}, label="Suggest Enrichment", icon="plug"),
        ],
    ).send()


@cl.action_callback("suggest_enrichment")
async def suggest_enrichment_action(action: cl.Action):
    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")

    mapping = cl.user_session.get("mapping") or {}

    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return

    if not cl.user_session.get("mapping_ready", False):
        sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, mapping)
        cl.user_session.set("mapping", sanitized_mapping)
        cl.user_session.set("pending_mapping_target", "enrichment")

        warning_text = ""
        if mapping_warnings:
            warning_text = "\n\nTemporal mapping guard:\n" + "\n".join([f"- {w}" for w in mapping_warnings])

        await cl.Message(
            content=(
                "Before suggesting enrichment, confirm the temporal mapping.\n\n"
                "Suggested mapping:\n"
                f"{_mapping_summary_markdown(sanitized_mapping)}\n\n"
                "You can override mapping if needed, then confirm."
                f"{warning_text}"
            ),
            actions=[
                cl.Action(
                    name="apply_mapping_override",
                    payload={
                        "command": (
                            "map "
                            f"paid_at={(sanitized_mapping or {}).get('paid_at') or ''} "
                            f"arrival_at={(sanitized_mapping or {}).get('arrival_at') or ''} "
                            f"expected_arrival_at={(sanitized_mapping or {}).get('expected_arrival_at') or ''}"
                        ).strip()
                    },
                    label="Override Mapping",
                    icon="wrench",
                ),
                cl.Action(name="skip_mapping_for_enrichment", payload={"action": "skip"}, label="Skip", icon="skip-forward"),
            ],
        ).send()
        return

    connected_sources = set(cl.user_session.get("connected_datasources") or [])
    suggestion = await recommend_enrichment(df, mapping, "explain delays", "internal explanation requested")
    if not suggestion:
        await cl.Message(content="No external enrichment is strongly justified right now.").send()
        return

    if suggestion.get("source_id") in connected_sources:
        await cl.Message(content=f"Datasource `{suggestion['source_id']}` is already connected.").send()
        return

    await cl.Message(
        content=(
            "Recommended enrichment:\n\n"
            f"- Source: `{suggestion['source_id']}`\n"
            f"- Why: {suggestion['reason']}\n"
            f"- Confidence: {suggestion['confidence']:.2f}"
        ),
        actions=[
            cl.Action(
                name="connect_datasource",
                payload={"source_id": suggestion["source_id"]},
                label=f"Connect {suggestion['source_id']}",
                icon="plug",
            )
        ],
    ).send()


@cl.action_callback("confirm_mapping_for_enrichment")
async def confirm_mapping_for_enrichment(action: cl.Action):
    cl.user_session.set("mapping_ready", True)
    cl.user_session.set("pending_mapping_target", None)
    await cl.Message(content="Mapping confirmed for enrichment suggestions.").send()
    await suggest_enrichment_action(cl.Action(name="suggest_enrichment", payload={"reason": "mapping_confirmed"}, label="Suggest Enrichment"))


@cl.action_callback("skip_mapping_for_enrichment")
async def skip_mapping_for_enrichment(action: cl.Action):
    cl.user_session.set("mapping_ready", True)
    cl.user_session.set("pending_mapping_target", None)
    await cl.Message(content="Proceeding with current mapping for enrichment suggestions.").send()
    await suggest_enrichment_action(cl.Action(name="suggest_enrichment", payload={"reason": "mapping_skipped"}, label="Suggest Enrichment"))


@cl.action_callback("confirm_mapping_for_insights")
async def confirm_mapping_for_insights(action: cl.Action):
    cl.user_session.set("mapping_ready", True)
    cl.user_session.set("pending_mapping_target", None)
    await cl.Message(content="Mapping confirmed for insights.").send()
    await generate_insights(cl.Action(name="generate_insights", payload={"reason": "mapping_confirmed"}, label="Generate Insights"))


@cl.action_callback("skip_mapping_for_insights")
async def skip_mapping_for_insights(action: cl.Action):
    cl.user_session.set("mapping_ready", True)
    cl.user_session.set("pending_mapping_target", None)
    await cl.Message(content="Proceeding with current mapping for insights.").send()
    await generate_insights(cl.Action(name="generate_insights", payload={"reason": "mapping_skipped"}, label="Generate Insights"))


@cl.action_callback("connect_datasource")
async def connect_datasource_action(action: cl.Action):
    source_id = ((action.payload or {}).get("source_id") or "").strip().lower()
    if not source_id:
        await cl.Message(content="Missing datasource id.").send()
        return

    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")
    mapping = cl.user_session.get("mapping") or {}
    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return

    try:
        async with cl.Step(name=f"Connect datasource: {source_id}") as step:
            step.output = "Fetching external data and enriching your dataset..."
            enriched, meta = await asyncio.to_thread(connect_datasource, df, mapping, source_id)
            cl.user_session.set("df_enriched", enriched)
            already_connected = set(cl.user_session.get("connected_datasources") or [])
            already_connected.add(source_id)
            cl.user_session.set("connected_datasources", sorted(already_connected))

            previous_holidays_connected = bool(cl.user_session.get("holidays_connected"))
            cl.user_session.set("holidays_connected", previous_holidays_connected or source_id == "openholidays")
            step.output = "Datasource connected. Recomputing derived metrics..."
            derived = await asyncio.to_thread(ensure_derived, enriched, mapping)
            cl.user_session.set("df_enriched", derived)
            step.output = "Datasource enrichment complete."
    except Exception as exc:
        await cl.Message(content=f"Could not connect datasource `{source_id}`: {exc}").send()
        return

    if source_id == "openholidays" and "delay_hours" in derived.columns and "is_bank_holiday" in derived.columns:
        holiday_delay = derived.loc[derived["is_bank_holiday"] & derived["delay_hours"].notna(), "delay_hours"]
        non_holiday_delay = derived.loc[(~derived["is_bank_holiday"]) & derived["delay_hours"].notna(), "delay_hours"]

        holiday_count = int(holiday_delay.shape[0])
        non_holiday_count = int(non_holiday_delay.shape[0])

        holiday_avg_text = f"{holiday_delay.mean():.2f}h" if holiday_count > 0 else "N/A (no matched holiday payouts)"
        non_avg_text = f"{non_holiday_delay.mean():.2f}h" if non_holiday_count > 0 else "N/A"

        holidays_rows = int(meta.get("holidays_rows", 0) or 0)

        await cl.Message(
            content=(
                f"Connected `{source_id}` ({meta.get('base_url')}).\n\n"
                f"- Avg delay on holidays: **{holiday_avg_text}**\n"
                f"- Avg delay on non-holidays: **{non_avg_text}**\n"
                f"- Holiday-matched payouts: **{holiday_count}**\n"
                f"- Holiday rows fetched: **{holidays_rows}**\n"
                f"- Countries processed: **{meta.get('country_count', 0)}**"
            ),
            actions=[
                cl.Action(
                    name="ask_followup",
                    payload={"q": "show top delays with holiday names"},
                    label="Show top delays with holiday names",
                    icon="pin",
                )
            ],
        ).send()
        return

    await cl.Message(content=f"Connected `{source_id}` and enriched your dataset.").send()


@cl.action_callback("ask_followup")
async def ask_followup(action: cl.Action):
    q = (action.payload or {}).get("q")

    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")

    mapping = cl.user_session.get("mapping") or {}
    sanitized_mapping, mapping_warnings = validate_temporal_mapping(df, mapping) if df is not None else (mapping, [])
    if sanitized_mapping != mapping:
        cl.user_session.set("mapping", sanitized_mapping)
    mapping = sanitized_mapping

    q_norm = (q or "").strip().lower()
    used_ai_deep_analysis = False

    if q_norm == "explain holiday impact":
        if df is None:
            await cl.Message(content="Upload a dataset first.").send()
            return

        async with cl.Step(name="Run follow-up") as step:
            step.output = "Analyzing holiday impact from enriched data..."
            response = await ai_explain_holiday_impact(df, mapping)
            step.output = "Follow-up ready."

        await cl.Message(content=f"**Follow-up:** {q}\n\n{response}").send()
        return

    async with cl.Step(name="Run follow-up") as step:
        step.output = "Analyzing follow-up question..."
        response = await asyncio.to_thread(answer_question, df, mapping, q)
        if should_use_llm_for_question(q or "", response or ""):
            step.output = "Generating deeper AI follow-up answer..."
            try:
                response = await ai_answer_question(df, mapping, q or "", response or "")
                used_ai_deep_analysis = True
            except Exception:
                print("AI follow-up QA fallback error:\n", traceback.format_exc())
        step.output = "Follow-up ready."

    followup_actions = None
    if q_norm == "show top delays with holiday names":
        followup_actions = [
            cl.Action(
                name="ask_followup",
                payload={"q": "explain holiday impact"},
                label="Explain holiday impact",
                icon="brain",
            )
        ]

    mapping_warning_text = ""
    if mapping_warnings:
        mapping_warning_text = "\n\n_Temporal mapping guard:_\n" + "\n".join([f"- {w}" for w in mapping_warnings])

    followup_content = f"**Follow-up:** {q}\n\n{response}{mapping_warning_text}"
    if used_ai_deep_analysis:
        followup_content = f"_AI deep analysis_\n\n{followup_content}"

    await cl.Message(
        content=followup_content,
        actions=followup_actions,
    ).send()

    await _suggest_enrichment_if_relevant(df, mapping, q or "", response or "")
