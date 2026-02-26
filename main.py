"""
=======================================================================
  main.py  —  SRE Log Analyzer (Gemini version)
=======================================================================

WHAT CHANGED FROM THE ANTHROPIC VERSION:

  OLD (Anthropic):
    import anthropic
    client = anthropic.Anthropic(api_key=...)
    client.messages.stream(model="claude-opus-4-5", ...)

  NEW (Google Gemini):
    import google.generativeai as genai
    genai.configure(api_key=...)
    model = genai.GenerativeModel("gemini-1.5-flash")
    model.generate_content(..., stream=True)

  Everything else — FastAPI, Pydantic, New Relic, streaming,
  the frontend — stays exactly the same.
  This is the power of clean architecture: swap one piece,
  rest of the system doesn't care.

HOW TO GET YOUR GEMINI API KEY:
  1. Go to https://aistudio.google.com
  2. Click "Get API key" → "Create API key"
  3. Copy it (starts with "AIza...")
  4. Set: export GEMINI_API_KEY="AIza..."

GEMINI MODELS AVAILABLE (free tier):
  gemini-1.5-flash  → Fastest, free, great for most tasks
  gemini-1.5-pro    → Smarter, free up to 2 req/min
  gemini-2.0-flash  → Latest, fast, free tier available

  We use gemini-1.5-pro by default — best quality for SRE analysis.
  Switch to gemini-1.5-flash if you want faster responses.
=======================================================================
"""

import os
import json
import hmac
import hashlib
from datetime import datetime
from typing import AsyncGenerator

# ── Gemini import ─────────────────────────────────────────────────────
# pip install google-generativeai
import google.generativeai as genai

# FastAPI imports (unchanged)
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# ── App setup (unchanged) ─────────────────────────────────────────────
app = FastAPI(
    title="SRE Log Analyzer",
    description="""
## AI-powered SRE tool — powered by Google Gemini

### Features
- **Log Analysis** — paste logs, get root cause + fix
- **New Relic Integration** — receive alerts via webhook, auto-analyzed by Gemini
- **Streaming** — responses appear word-by-word

### Endpoints
- `POST /analyze` — stream log analysis
- `POST /chat` — follow-up questions
- `POST /newrelic/simulate` — fire a test alert
- `POST /newrelic/webhook` — real New Relic alerts
- `GET  /health` — health check
    """,
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")

# ── Gemini client setup ───────────────────────────────────────────────
# This replaces: client = anthropic.Anthropic(api_key=...)
# genai.configure sets the API key globally for all Gemini calls
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Create model instances
# GenerativeModel is the equivalent of Anthropic's client
# We create two: one for log analysis, one for New Relic alerts
# This lets us give each a different "personality" via system_instruction

log_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",

    # system_instruction = Gemini's version of Anthropic's "system" prompt
    # This tells Gemini WHO it is and HOW to behave for all conversations
    system_instruction="""You are an expert SRE (Site Reliability Engineer) with 10 years of experience at top tech companies.
Analyze logs and always respond in this exact markdown format:

## 🔴 Root Cause
[One clear sentence explaining what went wrong]

## 📊 Severity
**[CRITICAL/HIGH/MEDIUM/LOW]** — [One sentence on business impact]

## 🔍 Key Evidence
[Bullet list of the specific log lines that prove your diagnosis]

## 🛠️ Immediate Fix
[Numbered step-by-step fix — be specific, include actual commands]

## 🛡️ Prevention
[2-3 concrete actions to prevent recurrence]

Be direct. No fluff. If asked follow-up questions, remember the logs you already analyzed."""
)

newrelic_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="""You are an expert SRE analyzing a New Relic alert.
Always respond in this exact markdown format:

## 🚨 Diagnosis
[One sentence: what is happening and why]

## 📊 Severity Assessment
**[CRITICAL/HIGH/MEDIUM/LOW]** — [User/business impact]

## 🔍 What New Relic Is Telling Us
[Plain English explanation of the metrics and thresholds]

## 🛠️ Immediate Actions
[Numbered steps with specific commands]

## 📈 Metrics to Watch
[Which New Relic metrics confirm the fix is working]

## 🛡️ Long-term Fix
[Permanent solution in 1-2 sentences]

Be specific. Include real commands. No fluff."""
)

# In-memory alert store
recent_alerts: list[dict] = []


# ── Pydantic models (unchanged) ───────────────────────────────────────

class LogAnalysisRequest(BaseModel):
    logs: str = Field(
        ..., min_length=10,
        description="Raw log text to analyze",
        example="2024-01-15 14:23:06 ERROR [payment-service] Connection timeout after 5000ms"
    )
    class Config:
        json_schema_extra = {"example": {"logs": "2024-01-15 ERROR [payment-service] DB connection timeout\n2024-01-15 ERROR All retries exhausted."}}


class ChatRequest(BaseModel):
    history: list[dict] = Field(default=[], description="Previous messages")
    question: str = Field(..., min_length=1, description="Follow-up question",
                          example="Give me the exact kubectl command to fix this")


class SimulateAlertRequest(BaseModel):
    scenario: str = Field(
        default="high_error_rate",
        description="Scenario: high_error_rate | slow_response | memory_leak | apdex_drop"
    )


# ── Gemini streaming helpers ──────────────────────────────────────────
#
# HOW GEMINI STREAMING DIFFERS FROM ANTHROPIC:
#
# Anthropic:
#   with client.messages.stream(...) as stream:
#       for chunk in stream.text_stream:
#
# Gemini:
#   response = model.generate_content(..., stream=True)
#   for chunk in response:
#       chunk.text  ← the text piece
#
# The concept is identical — we just call different methods.

def convert_history_to_gemini(history: list[dict]) -> list:
    """
    Convert our conversation history format to what Gemini expects.

    Our format (same as Anthropic/OpenAI):
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Gemini format:
      [{"role": "user", "parts": ["..."]}, {"role": "model", "parts": ["..."]}]

    Key differences:
      - "content" → "parts" (and it's a list)
      - "assistant" → "model" (Gemini calls it "model" not "assistant")
    """
    gemini_history = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({
            "role": role,
            "parts": [msg["content"]]
        })
    return gemini_history


async def stream_gemini_response(
    prompt: str,
    model: genai.GenerativeModel,
    history: list[dict] = None
) -> AsyncGenerator[str, None]:
    """
    Stream Gemini's response as Server-Sent Events.

    If history is provided, Gemini gets conversation context.
    If not, it's a fresh single-turn conversation.
    """
    try:
        if history:
            # Multi-turn conversation: start a chat session with history
            # This is Gemini's way of maintaining conversation context
            gemini_history = convert_history_to_gemini(history[:-1])  # all but last
            chat = model.start_chat(history=gemini_history)
            # Send the last message (the new question)
            last_message = history[-1]["content"]
            response = chat.send_message(last_message, stream=True)
        else:
            # Single-turn: just generate from the prompt
            response = model.generate_content(prompt, stream=True)

        full_reply = ""
        for chunk in response:
            # chunk.text contains the text piece (may be empty for some chunks)
            if chunk.text:
                full_reply += chunk.text
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"

        yield f"data: {json.dumps({'done': True, 'full_reply': full_reply})}\n\n"

    except Exception as e:
        error_msg = f"\n\n⚠️ Gemini error: {str(e)}"
        yield f"data: {json.dumps({'text': error_msg})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full_reply': error_msg})}\n\n"


def call_gemini_sync(prompt: str, model: genai.GenerativeModel) -> str:
    """
    Non-streaming Gemini call for webhook processing.
    Returns the full response text.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini analysis failed: {e}"


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post(
    "/analyze",
    summary="Analyze logs with Gemini AI",
    description="Send raw log text. Streams back root cause analysis word-by-word.",
    tags=["Log Analysis"]
)
async def analyze(body: LogAnalysisRequest):
    prompt = f"Analyze these logs and find the root cause:\n\n{body.logs}"

    return StreamingResponse(
        stream_gemini_response(prompt, log_model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post(
    "/chat",
    summary="Follow-up questions with conversation memory",
    description="Ask follow-up questions. Pass conversation history so Gemini remembers context.",
    tags=["Log Analysis"]
)
async def chat(body: ChatRequest):
    # Build full message list including the new question
    messages = body.history + [{"role": "user", "content": body.question}]

    return StreamingResponse(
        stream_gemini_response(body.question, log_model, history=messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── New Relic integration (logic unchanged, just swapped AI call) ─────

def verify_signature(body: bytes, header: str) -> bool:
    secret = os.environ.get("NEWRELIC_WEBHOOK_SECRET", "")
    if not secret:
        return True
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", header or "")


def format_alert_prompt(alert: dict) -> str:
    thresholds = "\n".join(
        f"  - {t.get('name','?')}: {t.get('value','?')} "
        f"(threshold: {t.get('threshold','?')} for {t.get('duration','?')} min)"
        for t in alert.get("thresholds", [])
    ) or "  No threshold data"

    return f"""New Relic Alert Fired:

Alert:      {alert.get('alertName', 'Unknown')}
Policy:     {alert.get('policyName', 'Unknown')}
Entity:     {alert.get('entityName', 'Unknown')} ({alert.get('entityType', '')})
Priority:   {alert.get('priority', 'Unknown')}
State:      {alert.get('currentState', 'Unknown')}
Time:       {alert.get('openTime', 'Unknown')}

Violated Thresholds:
{thresholds}

Current value:   {alert.get('metricValue', 'N/A')}
Threshold value: {alert.get('thresholdValue', 'N/A')}

Context: {alert.get('details', 'None provided.')}"""


def store_alert(alert: dict, diagnosis: str, source: str) -> dict:
    record = {
        "id": f"{source[:3]}-{len(recent_alerts)+1:04d}",
        "received_at": datetime.utcnow().strftime("%H:%M:%S UTC"),
        "source": source,
        "alert_name": alert.get("alertName", "Unknown"),
        "policy": alert.get("policyName", ""),
        "entity": alert.get("entityName", "Unknown"),
        "priority": alert.get("priority", "MEDIUM").upper(),
        "state": alert.get("currentState", "open"),
        "metric_value": alert.get("metricValue", "N/A"),
        "threshold_value": alert.get("thresholdValue", "N/A"),
        "ai_diagnosis": diagnosis,
        "raw_alert": alert,
    }
    recent_alerts.insert(0, record)
    if len(recent_alerts) > 20:
        recent_alerts.pop()
    return record


SCENARIOS = {
    "high_error_rate": {
        "alertName": "High Error Rate — payment-service",
        "policyName": "Production SLO Policy",
        "conditionName": "Error rate exceeds 5%",
        "entityName": "payment-service", "entityType": "APPLICATION",
        "currentState": "open", "previousState": "closed", "priority": "CRITICAL",
        "metricValue": "18.3", "thresholdValue": "5.0",
        "accountName": "Acme Corp Production",
        "alertUrl": "https://alerts.newrelic.com/accounts/123/incidents/456",
        "details": "Error rate spiked to 18.3% (threshold: 5%). Began ~4 min ago. Recent deploy: v2.4.1 deployed 45 min ago.",
        "thresholds": [{"name": "Error Rate", "value": "18.3%", "operator": ">", "threshold": "5%", "duration": "5"}]
    },
    "slow_response": {
        "alertName": "Latency Degradation — api-gateway",
        "policyName": "Production SLO Policy",
        "conditionName": "p99 response time exceeds 2000ms",
        "entityName": "api-gateway", "entityType": "APPLICATION",
        "currentState": "open", "previousState": "closed", "priority": "HIGH",
        "metricValue": "4230", "thresholdValue": "2000",
        "accountName": "Acme Corp Production",
        "alertUrl": "https://alerts.newrelic.com/accounts/123/incidents/457",
        "details": "p99 latency 4230ms (2x threshold). p50 normal at 890ms. postgres-primary query latency up from 12ms to 3400ms p99.",
        "thresholds": [{"name": "Response Time p99", "value": "4230ms", "operator": ">", "threshold": "2000ms", "duration": "3"}]
    },
    "memory_leak": {
        "alertName": "Memory Usage Critical — order-service",
        "policyName": "Infrastructure Policy",
        "conditionName": "Host memory exceeds 90%",
        "entityName": "order-service-pod-7d9f8b", "entityType": "HOST",
        "currentState": "open", "previousState": "closed", "priority": "HIGH",
        "metricValue": "94.2", "thresholdValue": "90.0",
        "accountName": "Acme Corp Production",
        "alertUrl": "https://alerts.newrelic.com/accounts/123/incidents/458",
        "details": "Memory grew steadily 62% → 94.2% over 90 min, no traffic increase. Classic memory leak. GC pauses increasing. Pod uptime: 6h.",
        "thresholds": [{"name": "Memory Used %", "value": "94.2%", "operator": ">", "threshold": "90%", "duration": "10"}]
    },
    "apdex_drop": {
        "alertName": "Apdex Score Drop — checkout-service",
        "policyName": "User Experience Policy",
        "conditionName": "Apdex score below 0.7",
        "entityName": "checkout-service", "entityType": "APPLICATION",
        "currentState": "open", "previousState": "closed", "priority": "CRITICAL",
        "metricValue": "0.41", "thresholdValue": "0.70",
        "accountName": "Acme Corp Production",
        "alertUrl": "https://alerts.newrelic.com/accounts/123/incidents/459",
        "details": "Apdex dropped 0.94 → 0.41. 59% of users degraded. Error rate 8.2%. Correlates with DB migration at 14:05 UTC.",
        "thresholds": [{"name": "Apdex Score", "value": "0.41", "operator": "<", "threshold": "0.70", "duration": "5"}]
    }
}


@app.post(
    "/newrelic/webhook",
    summary="Receive real New Relic alert webhook",
    description="Configure this URL in New Relic: Alerts → Channels → Webhook → paste this URL.",
    tags=["New Relic Integration"]
)
async def newrelic_webhook(
    request: Request,
    x_nr_webhook_signature: str = Header(default="")
):
    body = await request.body()
    if not verify_signature(body, x_nr_webhook_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        alert = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not alert:
        raise HTTPException(status_code=400, detail="Empty payload")

    # Only change: call_gemini_sync instead of run_claude_sync
    diagnosis = call_gemini_sync(format_alert_prompt(alert), newrelic_model)
    record = store_alert(alert, diagnosis, source="newrelic")

    return {"status": "received", "alert_id": record["id"]}


@app.post(
    "/newrelic/simulate",
    summary="Simulate a New Relic alert (demo)",
    description="Valid scenarios: high_error_rate | slow_response | memory_leak | apdex_drop",
    tags=["New Relic Integration"]
)
async def simulate_alert(body: SimulateAlertRequest):
    if body.scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario. Valid: {list(SCENARIOS.keys())}"
        )
    payload = {
        **SCENARIOS[body.scenario],
        "openTime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    diagnosis = call_gemini_sync(format_alert_prompt(payload), newrelic_model)
    record = store_alert(payload, diagnosis, source="simulated")
    return {"status": "simulated", "alert_id": record["id"]}


@app.get(
    "/newrelic/alerts",
    summary="Get recent alerts",
    tags=["New Relic Integration"]
)
async def get_alerts():
    return {"alerts": recent_alerts}


@app.get(
    "/newrelic/alerts/{alert_id}",
    summary="Get single alert by ID",
    tags=["New Relic Integration"]
)
async def get_alert(alert_id: str):
    alert = next((a for a in recent_alerts if a["id"] == alert_id), None)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    return alert


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "ai_provider": "Google Gemini",
        "model": "gemini-2.5-flash",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )
