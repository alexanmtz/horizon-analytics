# Horizon

AI Analytics Beyond the Data Horizon

## Problem

Most analytics tools assume that the data already contains the answers. In practice, however, many analytical questions require **context that exists outside the dataset**.

For example, when analyzing payout delays, a table may contain timestamps, countries, and providers. It can reveal that some transfers take longer than others, but it cannot explain why. The underlying cause may be external factors such as:

- bank holidays  
- regional banking systems  
- settlement windows  
- payment infrastructure differences  

## Why

Traditional analytics platforms focus on querying internal data. When external context is required, users must manually discover, fetch, and integrate those sources themselves.

I wanted to explore a different approach: **an analytics system that actively identifies missing context and suggests external knowledge sources during analysis.**

This led to the concept of **Horizon** — an AI-first analytics assistant designed to help users move beyond the “data horizon”.

---

## Workflow

A small **demo dataset is included in the repository** to illustrate the workflow:
https://github.com/alexanmtz/horizon-analytics/tree/main/sample_data

You can copy and paste or download and upload the data on the system.

1. The user uploads a dataset (e.g., a payouts table).
2. Horizon generates a structural overview of the data.
3. The user asks exploratory questions such as:
   - “What is the average transfer time?”
4. The AI detects anomalies or patterns:
   - “Some payouts take significantly longer than others.”
5. The AI proposes possible explanations:
   - country differences  
   - provider differences  
   - **bank holidays**
6. The system suggests enriching the dataset by connecting to an external source (e.g., a public holidays API).

The key idea is that insights can connect to external resources that can give additional insights.

> **Note:** The current implementation is a **working prototype / demo** intended to demonstrate the AI-driven analytical workflow. Most of the functionalities are limited and it's a proof of concept with minimal use cases.

---

## Key Design Decisions

### AI-First Interaction

Horizon is designed around a **conversation-driven analytical workflow**.

The AI handles:

- dataset interpretation
- exploratory analysis
- hypothesis generation
- suggesting next analytical steps

---

### Context Expansion

The system wants to allow the AI to suggest and integrate contextual data sources such as:

- public holiday calendars  
- economic indicators  
- geography and time zones  
- banking infrastructure  

This transforms analytics from simple **data exploration** into **context discovery**, as apply the context to extend the data.

---

### Lightweight Prototype Architecture

The prototype focuses on validating the AI-driven workflow rather than building a full analytics stack.

Main components:

- **Chainlit** for conversational UI
- **Python + Pandas** for data inspection and transformation
- **OpenAI models** for reasoning and hypothesis generation
- **External APIs** (e.g., holiday datasets) for contextual enrichment
- **Deploy** Deployed on https://render.com/

---

## What I Would Build Next

### Autonomous Hypothesis Generation

Currently, insights are driven by user questions. The next step would be for the system to proactively generate hypotheses such as:

- country-level anomalies  
- seasonality patterns  
- infrastructure-related delays  

This would turn Horizon into a **proactive analytics assistant**.

---

### External Data Discovery

Instead of manually integrating external APIs, Horizon could automatically discover and connect to relevant datasets such as:

- public open-data portals  
- economic indicators  
- financial infrastructure datasets  

This would allow the AI to continuously expand the analytical context.

---

### MCP-Based Tool Ecosystem

A natural evolution of this system would be integrating **Model Context Protocol (MCP)**.

Using MCP would allow Horizon to dynamically access external tools and data sources, such as:

- public datasets  
- APIs  
- internal services  
- knowledge bases  

This would enable a more agentic architecture where the AI can discover and use new capabilities without requiring hardcoded integrations.