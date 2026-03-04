import asyncio
import traceback
from dotenv import load_dotenv
import chainlit as cl

from helpers.followup_profile import followup_profile
from helpers.ai_brief_from_metrics import ai_brief_from_metrics
from helpers.ai_followups import suggest_followups
from helpers.anomaly import ai_explain_anomaly, detect_biggest_anomaly
from ingest import load_table, load_table_from_text
from profiling import profile_df
from semantic import suggest_mapping, apply_mapping_override
from metrics import compute_all_metrics
from qa import answer_question

from data_sources.enrichment_engine import connect_datasource
from helpers.ensure_derived import ensure_derived
from helpers.enrichment_recommender import recommend_enrichment

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("df", None)
    cl.user_session.set("mapping", {})
    await cl.Message(content="Upload a CSV/XLSX to begin or paste the data here.").send()


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

            step.output = "Inferring semantic column mapping..."
            mapping = await asyncio.to_thread(suggest_mapping, df)
            cl.user_session.set("mapping", mapping)

            step.output = f"Loaded {len(df)} rows from pasted data and prepared mapping."
        loaded_df = df
        loaded_mapping = mapping

    if loaded_df is not None:
        await cl.Message(
            content=(
                f"Loaded {len(loaded_df)} rows.\n\n"
                "Suggested mapping:\n"
                f"- arrival_at: `{(loaded_mapping or {}).get('arrival_at')}`\n"
                f"- expected_arrival_at: `{(loaded_mapping or {}).get('expected_arrival_at')}`\n\n"
                "Use the button below to apply a mapping override."
            ),
            actions=[
                cl.Action(
                    name="apply_mapping_override",
                    payload={"command": "map arrival_at=arrival_at"},
                    label="Override Mapping",
                    icon="wrench",
                ),
                cl.Action(name="skip_mapping", payload={"action": "skip"}, label="Skip", icon="skip-forward"),
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
                f"- arrival_at: `{new_mapping.get('arrival_at')}`\n"
                f"- expected_arrival_at: `{new_mapping.get('expected_arrival_at')}`\n\n"
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

    async with cl.Step(name="Answer question") as step:
        step.output = "Running analytics Q&A..."
        response = await asyncio.to_thread(answer_question, df, mapping, (msg.content or ""))

        step.output = "Generating suggested follow-up questions..."
        profile = await asyncio.to_thread(followup_profile, df, mapping)
        followups = await suggest_followups(msg.content or "", response, profile)

        step.output = "Q&A complete."

    # Send the main answer
    await cl.Message(content=response or "DEBUG: QA returned empty").send()

    suggestion = await recommend_enrichment(df, mapping, msg.content or "", response or "")
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
    cl.user_session.set("mapping", new_mapping)

    await cl.Message(
        content=(
            "Mapping updated.\n"
            f"- arrival_at: `{new_mapping.get('arrival_at')}`\n"
            f"- expected_arrival_at: `{new_mapping.get('expected_arrival_at')}`\n\n"
            "Now click Confirm."
        ),
        actions=[
            cl.Action(name="confirm_mapping", payload={"action": "confirm"}, label="Confirm", icon="check"),
        ],
    ).send()


@cl.action_callback("confirm_mapping")
async def confirm_mapping(action: cl.Action):

    await _generate_initial_insights(
        step_name="Confirm mapping",
        completion_step_output="Mapping confirmed and insights ready.",
        summary_prefix="Mapping confirmed.",
    )


@cl.action_callback("skip_mapping")
async def skip_mapping(action: cl.Action):
    await cl.Message(content="Keeping the data as is (no mapping override applied).", author="Assistant").send()

    await _generate_initial_insights(
        step_name="Skip mapping override",
        completion_step_output="Kept mapping as-is and prepared insights.",
        summary_prefix="Kept the data as is.",
    )


async def _generate_initial_insights(
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

    async with cl.Step(name=step_name) as step:
        step.output = "Preparing derived columns..."
        derived = await asyncio.to_thread(ensure_derived, df, mapping)
        cl.user_session.set("df_enriched", derived)

        step.output = "Computing metrics..."
        metrics_text = await asyncio.to_thread(compute_all_metrics, derived, mapping)

        step.output = "Generating AI insight brief..."
        brief = ""
        ai_task = None
        try:
            ai_task = asyncio.create_task(
                ai_brief_from_metrics(derived, mapping, metrics_text, timeout_seconds=15.0)
            )
            done, _ = await asyncio.wait({ai_task}, timeout=20)
            if done:
                brief = ai_task.result()
            else:
                brief = "_(AI brief timed out — showing metrics only.)_"
                ai_task.cancel()
        except Exception:
            brief = "_(AI brief failed — showing metrics only.)_"
            if ai_task and not ai_task.done():
                ai_task.cancel()
            print("AI brief error:\n", traceback.format_exc())

        step.output = completion_step_output

    await cl.Message(
        content=(
            f"{summary_prefix}\n\n"
            "### Initial Insight Brief\n\n"
            f"{brief}\n\n"
            f"{metrics_text}\n\n"
            "Ask a question to explore drivers. If needed, I can suggest external enrichments."
        ),
       actions=[
            cl.Action(name="suggest_enrichment", payload={"reason": "initial_insights"}, label="Suggest Enrichment", icon="plug"),
            cl.Action(name="explain_anomaly", payload={"action": "explain"}, label="Explain biggest anomaly", icon="zap"),
            cl.Action(
                name="ask_followup",
                payload={"q": "show top delays with holiday names"},
                label="Show top delays with holiday names",
                icon="pin",
            ),
            cl.Action(
                name="ask_followup",
                payload={"q": "explain holiday impact"},
                label="Explain holiday impact",
                icon="brain",
            ),
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

    suggestion = await recommend_enrichment(df, mapping, "explain delays", "internal explanation requested")
    if not suggestion:
        await cl.Message(content="No external enrichment is strongly justified right now.").send()
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


@cl.action_callback("connect_datasource")
async def connect_datasource_action(action: cl.Action):
    source_id = ((action.payload or {}).get("source_id") or "").strip().lower()
    if not source_id:
        await cl.Message(content="Missing datasource id.").send()
        return

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
            cl.user_session.set("holidays_connected", source_id == "openholidays")
            step.output = "Datasource connected. Recomputing derived metrics..."
            derived = await asyncio.to_thread(ensure_derived, enriched, mapping)
            cl.user_session.set("df_enriched", derived)
            step.output = "Datasource enrichment complete."
    except Exception as exc:
        await cl.Message(content=f"Could not connect datasource `{source_id}`: {exc}").send()
        return

    if source_id == "openholidays" and "delay_hours" in derived.columns and "is_bank_holiday" in derived.columns:
        holiday_avg = derived.loc[derived["is_bank_holiday"] & derived["delay_hours"].notna(), "delay_hours"].mean()
        non_avg = derived.loc[(~derived["is_bank_holiday"]) & derived["delay_hours"].notna(), "delay_hours"].mean()

        await cl.Message(
            content=(
                f"Connected `{source_id}` ({meta.get('base_url')}).\n\n"
                f"- Avg delay on holidays: **{holiday_avg:.2f}h**\n"
                f"- Avg delay on non-holidays: **{non_avg:.2f}h**\n"
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


@cl.action_callback("connect_holidays")
async def connect_holidays_backward_compat(action: cl.Action):
    action.payload = {"source_id": "openholidays"}
    await connect_datasource_action(action)

@cl.action_callback("explain_anomaly")
async def explain_anomaly(action: cl.Action):
    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")

    mapping = cl.user_session.get("mapping") or {}

    if df is None:
        await cl.Message(content="Upload a dataset first.").send()
        return

    async with cl.Step(name="Explain anomaly") as step:
        step.output = "Detecting biggest anomaly..."
        anomaly = await asyncio.to_thread(detect_biggest_anomaly, df, mapping)
    if anomaly is None:
        await cl.Message(content="I couldn't find an anomaly (missing delay/transfer columns).").send()
        return

    has_holidays = bool(cl.user_session.get("holidays_connected"))
    async with cl.Step(name="Explain anomaly") as step:
        step.output = "Generating AI explanation..."
        explanation = await ai_explain_anomaly(anomaly, mapping, has_holidays=has_holidays)
        step.output = "Anomaly explanation ready."

    await cl.Message(
        content=(
            "### Biggest anomaly detected\n\n"
            f"- Metric: **{anomaly['metric']}**\n"
            f"- Z-score: **{anomaly['z_score']:.2f}**\n"
            f"- Row: `{anomaly['row']}`\n\n"
            f"{explanation}"
        )
    ).send()

@cl.action_callback("ask_followup")
async def ask_followup(action: cl.Action):
    q = (action.payload or {}).get("q")

    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")

    mapping = cl.user_session.get("mapping") or {}

    async with cl.Step(name="Run follow-up") as step:
        step.output = "Analyzing follow-up question..."
        response = await asyncio.to_thread(answer_question, df, mapping, q)
        step.output = "Follow-up ready."

    followup_actions = None
    if (q or "").strip().lower() == "show top delays with holiday names":
        followup_actions = [
            cl.Action(
                name="ask_followup",
                payload={"q": "explain holiday impact"},
                label="Explain holiday impact",
                icon="brain",
            )
        ]

    await cl.Message(
        content=f"**Follow-up:** {q}\n\n{response}",
        actions=followup_actions,
    ).send()
