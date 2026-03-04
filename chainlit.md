# Horizon Analytics

AI-first analytics for payout and operations datasets.

## How to start

1. Upload a **CSV** or **XLSX** file, or paste tabular data (CSV/TSV with header).
2. Review the suggested semantic mapping (`arrival_at`, `expected_arrival_at`).
3. Confirm mapping to generate the initial metrics brief.
4. Ask follow-up questions to investigate delays, anomalies, and drivers.

Sample datasets:
https://github.com/alexanmtz/horizon-analytics/tree/main/sample_data

## What you can ask

- "What is the average delay and how is it trending?"
- "Show the biggest anomalies and explain likely causes."
- "Break down delays by transfer type and amount buckets."
- "Which factors most influence late arrivals?"
- "Show top delays with holiday names."

## Enrichment support

The assistant can recommend and connect external enrichments (for example, holidays via OpenHolidays) when useful.

After enrichment, you can ask:

- "Explain holiday impact"
- "Compare holiday vs non-holiday delay"

## Tips

- Use clear date and business context in questions for better answers.
- If the mapping looks wrong, use the **Override Mapping** action before confirming.
- You can always continue exploring with suggested follow-up actions in the chat.
