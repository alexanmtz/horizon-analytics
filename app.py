import asyncio
import os
import traceback
from dotenv import load_dotenv
import chainlit as cl

from helpers import followup_profile
from helpers.ai_brief_from_metrics import ai_brief_from_metrics
from helpers.ai_followups import suggest_followups
from helpers.anomaly import ai_explain_anomaly, detect_biggest_anomaly
from ingest import load_table
from profiling import profile_df
from semantic import suggest_mapping, apply_mapping_override
from metrics import compute_all_metrics
from qa import answer_question
import pandas as pd

from data_sources.holiday.holidays_client import fetch_holidays
from data_sources.holiday.enrich_holidays import enrich_with_holidays
from helpers.ensure_derived import ensure_derived

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("df", None)
    cl.user_session.set("mapping", {})
    await cl.Message(content="Upload a CSV/XLSX to begin.").send()


@cl.on_message
async def on_message(msg: cl.Message):
    # 1) File upload
    if msg.elements:
        file_el = next((e for e in msg.elements if hasattr(e, "path")), None)
        if not file_el:
            await cl.Message(content="No file detected. Upload CSV/XLSX.").send()
            return

        async with cl.Step(name="Process upload") as step:
            step.output = "Reading uploaded file..."
            df = await asyncio.to_thread(load_table, file_el.path)
            cl.user_session.set("df", df)

            step.output = "Inferring semantic column mapping..."
            mapping = await asyncio.to_thread(suggest_mapping, df)
            cl.user_session.set("mapping", mapping)

            step.output = f"Loaded {len(df)} rows and prepared mapping."

        await cl.Message(
            content=(
                f"✅ Loaded {len(df)} rows.\n\n"
                f"Suggested mapping:\n"
                f"- arrival_at: `{mapping.get('arrival_at')}`\n"
                f"- expected_arrival_at: `{mapping.get('expected_arrival_at')}`\n\n"
                "Use the button below to apply a mapping override."
            ),
            actions=[
                cl.Action(
                    name="apply_mapping_override",
                    payload={"command": "map arrival_at=arrival_at"},
                    label="🛠 Override Mapping",
                ),
                cl.Action(name="skip_mapping", payload={
                          "action": "skip"}, label="⏭ Skip"),
                cl.Action(name="show_profile", payload={
                          "action": "profile"}, label="📊 View Profile"),
            ],
        ).send()
        return

    # 2) Commands
    text = (msg.content or "").strip()
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
                "✅ Mapping updated.\n"
                f"- arrival_at: `{new_mapping.get('arrival_at')}`\n"
                f"- expected_arrival_at: `{new_mapping.get('expected_arrival_at')}`\n\n"
                "Now click ✅ Confirm."
            ),
            actions=[
                cl.Action(name="confirm_mapping", payload={
                          "action": "confirm"}, label="✅ Confirm"),
            ],
        ).send()
        return

    # 3) Normal Q&A
    df = cl.user_session.get("df_enriched")
    if df is None:
        df = cl.user_session.get("df")

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

    # Show clickable followups
    await cl.Message(
        content="### Suggested next questions",
        actions=[
            cl.Action(name="ask_followup", payload={"q": q}, label=q)
            for q in followups
        ],
    ).send()

    return


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
            "✅ Mapping updated.\n"
            f"- arrival_at: `{new_mapping.get('arrival_at')}`\n"
            f"- expected_arrival_at: `{new_mapping.get('expected_arrival_at')}`\n\n"
            "Now click ✅ Confirm."
        ),
        actions=[
            cl.Action(name="confirm_mapping", payload={"action": "confirm"}, label="✅ Confirm"),
        ],
    ).send()


@cl.action_callback("confirm_mapping")
async def confirm_mapping(action: cl.Action):

    await _generate_initial_insights(
        step_name="Confirm mapping",
        completion_step_output="Mapping confirmed and insights ready.",
        summary_prefix="✅ Mapping confirmed.",
    )


@cl.action_callback("skip_mapping")
async def skip_mapping(action: cl.Action):
    await cl.Message(content="⏭ Keeping the data as is (no mapping override applied).", author="Assistant").send()

    await _generate_initial_insights(
        step_name="Skip mapping override",
        completion_step_output="Kept mapping as-is and prepared insights.",
        summary_prefix="⏭ Kept the data as is.",
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
            "If you want explanations for calendar-driven delays, connect a holiday calendar:"
        ),
       actions=[
            cl.Action(name="connect_holidays", payload={"source": "openholidays"}, label="🔌 Connect Holiday Calendar"),
            cl.Action(name="explain_anomaly", payload={"action": "explain"}, label="⚡ Explain biggest anomaly"),
            cl.Action(
                name="ask_followup",
                payload={"q": "show top delays with holiday names"},
                label="📌 Show top delays with holiday names",
            ),
            cl.Action(
                name="ask_followup",
                payload={"q": "explain holiday impact"},
                label="🧠 Explain holiday impact",
            ),
        ],
    ).send()


@cl.action_callback("connect_holidays")
async def connect_holidays(action: cl.Action):
    df = cl.user_session.get("df")
    mapping = cl.user_session.get("mapping") or {}
    if df is None:
        await cl.Message(content="No dataset loaded.").send()
        return

    async with cl.Step(name="Connect holidays") as step:
        step.output = "Validating dataset columns and mapping..."

        # Determine countries + date range from dataset
        if "country" not in df.columns:
            await cl.Message(content="Your dataset has no `country` column, so I can’t join holidays.").send()
            return

        exp_col = mapping.get("expected_arrival_at")
        if not exp_col or exp_col not in df.columns:
            await cl.Message(content="Missing `expected_arrival_at` mapping.").send()
            return

        exp = pd.to_datetime(df[exp_col], errors="coerce", utc=True)
        start_date = exp.min().strftime("%Y-%m-%d")
        end_date = exp.max().strftime("%Y-%m-%d")

        countries = sorted(
            [c for c in df["country"].dropna().unique().tolist() if isinstance(c, str)])

        step.output = f"Fetching holidays for {len(countries)} countries ({start_date} to {end_date})..."

        # Fetch from OpenHolidays
        all_h = []
        for idx, c in enumerate(countries, start=1):
            try:
                all_h.append(await asyncio.to_thread(fetch_holidays, c, start_date, end_date))
                step.output = f"Fetched {idx}/{len(countries)} countries..."
            except Exception:
                step.output = f"Skipped {c} due to API error ({idx}/{len(countries)})."
                continue

        holidays_df = pd.concat(all_h, ignore_index=True) if all_h else pd.DataFrame(
            columns=["country", "date", "name"])
        cl.user_session.set("holidays_df", holidays_df)

        step.output = "Enriching payouts with holiday flags..."
        enriched = await asyncio.to_thread(enrich_with_holidays, df, holidays_df, mapping)
        cl.user_session.set("df_enriched", enriched)
        cl.user_session.set("holidays_connected", True)

        step.output = "Computing holiday delay comparison..."

        # Quick explanation metrics
        from metrics import add_derived_columns
        derived = await asyncio.to_thread(add_derived_columns, enriched, mapping)
        derived["is_bank_holiday"] = enriched.get("is_bank_holiday", False)

    if "delay_hours" in derived.columns:
        holiday_avg = derived.loc[derived["is_bank_holiday"] &
                                  derived["delay_hours"].notna(), "delay_hours"].mean()
        non_avg = derived.loc[(~derived["is_bank_holiday"]) &
                              derived["delay_hours"].notna(), "delay_hours"].mean()

        await cl.Message(
            content=(
                "✅ Connected OpenHolidaysAPI and enriched payouts.\n\n"
                f"- Avg delay on bank holidays: **{holiday_avg:.2f}h**\n"
                f"- Avg delay on non-holidays: **{non_avg:.2f}h**\n\n"
            ),
            actions=[
                cl.Action(
                    name="ask_followup",
                    payload={"q": "show top delays with holiday names"},
                    label="📌 Show top delays with holiday names",
                ),
            ],
        ).send()
    else:
        await cl.Message(content="Connected holidays, but delay_hours is missing.").send()

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
            "### ⚡ Biggest anomaly detected\n\n"
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
                label="🧠 Explain holiday impact",
            )
        ]

    await cl.Message(
        content=f"**Follow-up:** {q}\n\n{response}",
        actions=followup_actions,
    ).send()
