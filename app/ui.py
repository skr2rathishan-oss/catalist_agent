"""Gradio dashboard: chat first, NVIDIA model selection below, live rate-limit visibility."""

from html import escape
import json
from queue import Empty, Queue
import re
import sqlite3
from threading import Thread
import time

from app.agents.qa_agent import QuestionAnsweringAgent
from app.config import ALL_TEAMS, BM25_INDEX_FILE, VECTOR_INDEX_FILE, PARSED_DIR, MEMORY_DIR, load_generation_settings
from app.generation import GenerationError
from app.llm import EmbeddingError, NvidiaEmbeddingClient
from app.retriever import search_hybrid, search_keywords, validate_search
from app.vector_store import search_vectors
from app.providers import (GenerationRouter, RATE_LIMIT_STATUSES, create_client, discover_models,
                           key_status, probe_model, usage_summary)

PROVIDER = "nvidia"
PASSED = "JSON check passed"
USAGE_HEADERS = ["State", "Kind", "Model", "Requests", "OK", "Rate limited",
                 "Last request (UTC)", "Last rate limit (UTC)"]


def readiness() -> str:
    """Show available inputs without reading transcript contents or making calls."""

    return (f"Keyword index: {'ready' if BM25_INDEX_FILE.is_file() else 'missing'} · "
            f"Vector index: {'ready' if VECTOR_INDEX_FILE.is_file() else 'missing'} · "
            f"Parsed meetings: {len(list(PARSED_DIR.glob('*.json')))}")


def alert_html(alerts) -> str:
    """Render rate-limit / fallback notices as a visible banner ("" when none)."""

    styles = {"error": ("rgba(220, 38, 38, 0.12)", "#dc2626", "Rate limit reached"),
              "warning": ("rgba(217, 119, 6, 0.12)", "#d97706", "Rate limit notice"),
              "info": ("rgba(37, 99, 235, 0.10)", "#2563eb", "Notice")}
    blocks = []
    for alert in alerts:
        background, border, default_title = styles.get(alert.get("level"), styles["info"])
        blocks.append(
            f'<div role="alert" style="background:{background};border-left:4px solid {border};'
            f'border-radius:6px;padding:10px 14px;margin:6px 0;">'
            f'<strong style="color:{border};">{escape(alert.get("title") or default_title)}</strong>'
            f'<div style="margin-top:4px;">{escape(alert["text"])}</div></div>')
    return "".join(blocks)


def parse_backups(backups, primary):
    """Distinct NVIDIA backup model IDs, excluding the primary, at most two."""

    models = []
    for value in backups or ():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Invalid backup model selection; test and reselect it")
        if value != primary and value not in models:
            models.append(value)
    return models[:2]


def stream_question(question, team, cutoff, method, limit, rounds, agent_factory=None,
                    model=None, backups=()):
    """Stream each real step; isolate request state and model clients per run.

    Yields (status, answer, citations, activity, evidence, result, metrics, alert_html, steps).
    """

    events = Queue()
    activity, evidence, metrics, alerts = [], {}, {}, []
    started = time.monotonic()
    current = {"model": model}

    def worker():

        try:
            meeting_cutoff = "d" if cutoff == "all" else cutoff
            team_id = validate_search(question, team, meeting_cutoff, limit)
            if not question.strip():
                raise ValueError("Enter a question")
            if method not in ("bm25", "embedding", "hybrid"):
                raise ValueError("Choose a supported retrieval method")
            if agent_factory is not None:
                agent = agent_factory(events.put)
                result = agent.answer(question, team_id, meeting_cutoff)
            else:
                required = [BM25_INDEX_FILE] if method == "bm25" else [VECTOR_INDEX_FILE]
                if method == "hybrid":
                    required.append(BM25_INDEX_FILE)
                if not PARSED_DIR.is_dir() or any(not path.is_file() for path in required):
                    raise ValueError("Required data/indexes are missing. Run scripts/build.py --embeddings first, "
                                     "or scripts/build.py for BM25 only.")
                routes = [(PROVIDER, model)] + [(PROVIDER, name) for name in parse_backups(backups, model)]
                if len(routes) > 1:
                    generation = GenerationRouter(routes, on_event=events.put)
                else:
                    generation = create_client(PROVIDER, model)
                    generation.on_event = events.put
                current["model"] = generation.settings.model
                events.put({"stage": "model_selected", "details": {
                    "model": current["model"], "backups": [name for _, name in routes[1:]]}})
                embedding = NvidiaEmbeddingClient() if method != "bm25" else None
                searches = {"bm25": search_keywords,
                            "embedding": lambda *a, **kw: search_vectors(*a, **kw, client=embedding),
                            "hybrid": lambda *a, **kw: search_hybrid(*a, **kw, client=embedding)}
                agent = QuestionAnsweringAgent(generation, searches[method], max_rounds=rounds,
                                               limit=limit, on_event=events.put,
                                               memory_dir=MEMORY_DIR)
                try:
                    result = agent.answer(question, team_id, meeting_cutoff)
                finally:
                    current["model"] = generation.settings.model
                metrics.update(generation_model=generation.settings.model,
                               routes_used=[route["model"] for route in
                                            getattr(generation, "history", [{"model": current["model"]}])],
                               generation_requests=generation.api_requests, generation_cache_hits=generation.cache_hits,
                               embedding_requests=embedding.api_requests if embedding else 0,
                               embedding_cache_hits=embedding.cache_hits if embedding else 0)
            metrics["elapsed_seconds"] = round(time.monotonic() - started, 2)
            events.put({"stage": "finished", "result": result, "metrics": dict(metrics)})
        except (ValueError, OSError, sqlite3.DatabaseError, GenerationError, EmbeddingError) as error:
            message = str(error)
            # Errors from adapters are sanitized; redact the configured key as a final guard.
            try:
                secret = load_generation_settings().api_key
                if secret:
                    message = message.replace(secret, "[redacted]")
            except (ValueError, OSError):
                pass
            stage = "error"
            status = getattr(error, "status", None)
            if isinstance(error, GenerationError) and status in RATE_LIMIT_STATUSES:
                stage = "rate_limited"
                kind = "rate limit (HTTP 429)" if status == 429 else "quota / credit limit (HTTP 402)"
                message = (f"NVIDIA model {current['model'] or '(selected)'} hit its {kind}. "
                           + ("All selected backup models were also rate limited. " if backups else "")
                           + "Choose another model under Model selection "
                             "or wait for the quota to reset.")
                delay = getattr(error, "retry_after", None)
                if delay and str(delay).isdigit():
                    message += f" NVIDIA asked to retry after {int(delay)} seconds."
            elif isinstance(error, EmbeddingError) and status == 429:
                stage = "rate_limited"
                message = ("NVIDIA embedding rate limit reached (used by hybrid/embedding search). "
                           "Switch Search method to bm25 to avoid embedding calls, or wait for the quota to reset.")
            events.put({"stage": stage, "message": message})
        except Exception:
            events.put({"stage": "error", "message": "Unexpected application error. Check the local runtime and tests."})

    steps = []  # Structured observable steps, rendered as boxes inside the chat.
    yield "Starting", "", [], "Starting agent…", {}, {}, {}, "", []
    Thread(target=worker, daemon=True).start()
    while True:
        try:
            event = events.get(timeout=0.25)
        except Empty:
            continue
        stage = event["stage"]
        if stage == "finished":
            result = event["result"]
            activity.append(f"Completed: {result['status']}")
            yield (result["status"], result.get("answer") or result.get("reason", "No supported answer"),
                   result.get("citations", []), "\n\n".join(activity), evidence, result, event["metrics"],
                   alert_html(alerts), list(steps))
            return
        if stage in ("error", "rate_limited"):
            activity.append(event["message"])
            if stage == "rate_limited":
                alerts.append({"level": "error", "text": event["message"]})
            yield (stage, event["message"], [], "\n\n".join(activity), evidence, {}, {}, alert_html(alerts),
                   list(steps))
            return
        details = event.get("details", {})
        steps.append({"stage": stage, "details": details})
        if stage == "rate_limit_retry":
            alerts.append({"level": "warning", "title": "Rate limited — retrying",
                           "text": f"{details['model']} returned HTTP 429. Waiting {details['wait_seconds']}s "
                                   f"before attempt {details['attempt'] + 1} of {details['max_attempts']}."})
        elif stage == "model_fallback":
            source, target = details["from"], details["to"]
            current["model"] = target["model"]
            alerts.append({"level": "warning", "title": "Rate limit — switched to backup model",
                           "text": f"{source['model']} was rate limited (HTTP 429); "
                                   f"continuing with {target['model']}."})
        if stage == "sources_ready":
            evidence["sources"] = details["sources"]
            evidence["citation_options"] = details["citation_options"]
            summary = {"source_turns": len(details["sources"]), "quote_options": len(details["citation_options"])}
        elif stage == "retrieved":
            evidence.setdefault("searches", []).append(details)
            summary = {key: value for key, value in details.items() if key != "chunks"}
            summary["chunk_ids"] = [chunk["chunk_id"] for chunk in details["chunks"]]
        else:
            summary = details
        activity.append(stage.replace("_", " ").title() + "\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        yield (stage.replace("_", " ").title(), "", [], "\n\n".join(activity), dict(evidence), {}, {},
               alert_html(alerts), list(steps))


MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~])")
CITATION_ID = re.compile(r"\s*\[[A-Z]{2}\d{4}[a-d]_t\d+_w\d+\]")
WAITING = {"model_selected": "🧭 Planning the question…", "planning": "🧭 Planning the question…",
           "plan_ready": "🔎 Searching the transcripts…", "coverage_repair": "✅ Re-checking the evidence…",
           "draft_repair": "✍️ Redrafting the answer…", "answer_verified": "🏁 Finishing…",
           "rate_limit_retry": "⚠️ Waiting for the rate limit…", "model_fallback": "🔁 Continuing on the backup model…",
           "searching": "🔎 Searching the transcripts…", "retrieved": "🔎 Searching the transcripts…",
           "sources_ready": "✅ Checking the evidence covers every part…",
           "checking_coverage": "✅ Checking the evidence covers every part…",
           "coverage_checked": "✍️ Drafting the answer…", "drafting": "✍️ Drafting the answer…",
           "verifying_answer": "🔍 Independent review of the answer…"}

PROCESS_CSS = """<style>
.rag-process{font-size:14px;line-height:1.45;color:var(--body-text-color,inherit)}
.rag-empty{color:var(--body-text-color-subdued,#6b7280);padding:8px 2px}
.rag-card{border:1px solid var(--border-color-primary,#d1d5db);border-radius:8px;padding:10px 12px;margin:0 0 8px;
  background:var(--background-fill-primary,transparent)}
.rag-card.pending{border-style:dashed;opacity:.85}
.rag-card.warn{border-left:4px solid #d97706}
.rag-card.fail{border-left:4px solid #dc2626}
.rag-card.ok{border-left:4px solid #16a34a}
.rag-head{display:flex;gap:8px;align-items:baseline;font-weight:600}
.rag-step{font-size:11px;font-weight:600;color:var(--body-text-color-subdued,#6b7280);min-width:44px}
.rag-body{margin-top:6px}
.rag-meta{font-size:12px;color:var(--body-text-color-subdued,#6b7280)}
.rag-chunk{border:1px solid var(--border-color-primary,#e5e7eb);border-radius:6px;margin:6px 0;padding:4px 8px}
.rag-chunk summary{cursor:pointer;font-weight:600}
.rag-text{white-space:pre-wrap;max-height:240px;overflow:auto;margin:6px 0 2px;padding:6px 8px;border-radius:4px;
  background:var(--background-fill-secondary,rgba(127,127,127,.08));font-size:13px}
.rag-quote{border-left:3px solid var(--color-accent,#6366f1);padding:2px 8px;margin:4px 0;white-space:pre-wrap}
.rag-badge{font-size:11px;font-weight:600;padding:1px 6px;border-radius:9px;background:rgba(22,163,74,.15);color:#16a34a}
.rag-process ul{margin:4px 0 0 18px;padding:0}
</style>"""


def md_escape(text) -> str:
    """Show transcript text literally (e.g. L_C_D_ stays as written)."""

    return MARKDOWN_SPECIAL.sub(r"\\\1", str(text))


def quote_block(text) -> str:
    return "\n".join("> " + md_escape(line) if line.strip() else ">" for line in str(text).splitlines())


def meeting_label(meeting_id) -> str:
    return f"{meeting_id} (team {meeting_id[:-1]}, meeting {meeting_id[-1]})"


def chunk_html(index, chunk, open_=False) -> str:
    """One retrieved chunk: source, why it was retrieved, and its original transcript text."""

    meta = []
    if chunk.get("start_timestamp"):
        meta.append(f"⏱ {chunk['start_timestamp']}–{chunk.get('end_timestamp', '?')}")
    if chunk.get("speaker_ids"):
        meta.append("Speakers " + ", ".join(chunk["speaker_ids"]))
    if chunk.get("turn_ids"):
        meta.append(f"turns {chunk['turn_ids'][0]}–{chunk['turn_ids'][-1]}")
    if isinstance(chunk.get("score"), (int, float)):
        meta.append(f"score {chunk['score']:.3f}")
    for method, value in (chunk.get("retrieval_scores") or {}).items():
        if isinstance(value, (int, float)):
            meta.append(f"{method} {value:.3f}")
    why = (f'<div class="rag-meta">Why retrieved: {escape(chunk["retrieval_reason"])}</div>'
           if chunk.get("retrieval_reason") else "")
    return (f'<details class="rag-chunk"{" open" if open_ else ""}><summary>{index}. '
            f'{escape(meeting_label(chunk["meeting_id"]))} · <code>{escape(chunk.get("chunk_id", "?"))}</code></summary>'
            f'<div class="rag-meta">{escape(" · ".join(meta))}</div>{why}'
            f'<div class="rag-text">{escape(chunk.get("original_text") or "")}</div></details>')


def quote_html(quote, source) -> str:
    return f'<div class="rag-quote">{escape(quote)}<div class="rag-meta">— {escape(source)}</div></div>'


def step_cards(steps):
    """Turn observable agent steps into (title, body_html, kind) cards, in order."""

    cards, queries = [], {}
    for step in steps:
        stage, details = step["stage"], step.get("details", {})
        if stage == "model_selected":
            backups = details.get("backups") or []
            cards.append(("🤖 Model selected", escape(f"{details.get('model')} · backups on HTTP 429: "
                                                     + (", ".join(backups) if backups else "none")), ""))
        elif stage == "plan_ready":
            parts = details.get("parts", [])
            items = "".join(f"<li><b>{escape(part['id'])}</b> {escape(part['question'])}"
                            f'<div class="rag-meta">search query: {escape(part["query"])}</div>'
                            + "".join(f'<div class="rag-meta">{escape(item["id"])}: {escape(item["question"])}</div>'
                                      for item in part.get("requirements", []))
                            + "</li>" for part in parts)
            cards.append((f"🧭 Planning — question split into {len(parts)} part(s)", f"<ul>{items}</ul>", ""))
        elif stage == "searching":
            queries[(details.get("round"), details.get("part_id"))] = details.get("query", "")
        elif stage == "retrieved":
            chunks = details.get("chunks", [])
            admission = details.get("admission", {})
            dropped = admission.get("dropped", [])
            budget_info = (f'<div class="rag-meta">Context: {admission.get("words_after", 0)} / '
                           f'{admission.get("total_word_budget", 0)} source words · '
                           f'{len(dropped)} turn(s) skipped by word budget</div>') if admission else ""
            if dropped:
                budget_info += '<div class="rag-meta">Skipped: ' + escape(
                    ", ".join(f'{item["meeting_id"]} turn {item["turn_id"]}' for item in dropped)) + '</div>'
            if admission.get("admitted"):
                budget_info += '<div class="rag-meta">Added to model context: ' + escape(
                    ", ".join(f'{item["meeting_id"]} turn {item["turn_id"]}' for item in admission["admitted"])) + '</div>'
            query = queries.get((details.get("round"), details.get("part_id")), "")
            body = (f'<div class="rag-meta">{len(chunks)} chunk(s) retrieved · '
                    f'{details.get("source_turns_added", 0)} new source turn(s) added to the evidence</div>'
                    + budget_info
                    + "".join(chunk_html(i, chunk, open_=i == 1) for i, chunk in enumerate(chunks, 1))
                    if chunks else "No matching passages.")
            cards.append((f"🔎 Executing search (round {details.get('round')}, {details.get('part_id')}): “{query}”",
                          body, ""))
        elif stage == "sources_ready":
            sources = details.get("sources", [])
            by_meeting = {}
            for source in sources:
                by_meeting.setdefault(source["meeting_id"], []).append(source)
            latest = {}
            for meeting_id in by_meeting:
                latest[meeting_id[:-1]] = max(latest.get(meeting_id[:-1], meeting_id), meeting_id)
            items = "".join(
                f"<li>{escape(meeting_label(meeting_id))}: {len(turns)} turn(s)"
                + (' <span class="rag-badge">latest meeting</span>' if latest[meeting_id[:-1]] == meeting_id else "")
                + "</li>" for meeting_id, turns in sorted(by_meeting.items()))
            cards.append((f"📚 Supporting evidence — {len(sources)} original turn(s) from {len(by_meeting)} meeting(s)",
                          f"<ul>{items}</ul><div class=\"rag-meta\">Meetings run a → d; a later meeting can replace an "
                          f"earlier decision, so evidence is compared in order. "
                          f"{len(details.get('citation_options', []))} exact-quote options offered to the model.</div>",
                          ""))
        elif stage == "coverage_checked":
            parts = (details.get("coverage") or {}).get("parts", [])
            missing = details.get("missing_parts", [])
            body = ""
            for part in parts:
                if part.get("supported"):
                    body += f"<div><b>{escape(part['id'])}</b> ✓ supported</div>" + "".join(
                        quote_html(evidence["quote"], f"{evidence['meeting_id']} turn {evidence['turn_id']}")
                        for evidence in part.get("evidence", []))
                else:
                    body += (f"<div><b>{escape(part['id'])}</b> ✗ not yet supported → follow-up search: "
                             f"<i>{escape(part.get('next_query') or '')}</i></div>")
                body += "".join(f'<div class="rag-meta">{escape(item["id"])}: '
                                + ("evidence found" if item["evidence"] else "missing evidence") + '</div>'
                                for item in part.get("requirements", []))
            cards.append((f"✅ Coverage check (round {details.get('round')}) — every part supported" if not missing else
                          f"🔁 Coverage check (round {details.get('round')}) — missing {', '.join(missing)}, searching again",
                          body, "ok" if not missing else "warn"))
        elif stage == "coverage_repair":
            cards.append(("🛠 Coverage answer rejected by code checks — repairing", escape(str(details.get("error"))), "warn"))
        elif stage == "verifying_answer":
            cards.append((f"🔍 Draft {details.get('attempt')} — citations matched word for word", "".join(
                quote_html(cite["quote"], f"{cite['meeting_id']} turn {cite['turn_id']} · supports "
                                          f"{', '.join(cite.get('part_ids', []))}")
                for cite in details.get("citations", [])), ""))
        elif stage == "answer_verified":
            cards.append(("✔ Independent review approved the answer", "No issues found.", "ok"))
        elif stage == "draft_repair":
            cards.append((f"🛠 Draft {details.get('attempt')} rejected — repairing", escape(str(details.get("error"))), "warn"))
        elif stage == "rate_limit_retry":
            cards.append((f"⚠️ Rate limited — retrying in {details.get('wait_seconds')}s",
                          escape(f"{details.get('model')} returned HTTP 429."), "warn"))
        elif stage == "model_fallback":
            cards.append(("⚠️ Rate limit — switched to backup model",
                          escape(f"{details['from']['model']} → {details['to']['model']}"), "warn"))
    return cards


def current_activity(steps) -> str:
    """One short line for the chat box while the agent works."""

    last = steps[-1]["stage"] if steps else "planning"
    if last == "searching":
        return "🔎 Searching: “" + steps[-1]["details"].get("query", "") + "”…"
    return WAITING.get(last, "⏳ Working…")


def process_html(steps, running=False, outcome=None) -> str:
    """The live process panel: one card per step, appearing in order."""

    if not steps and not running and outcome is None:
        return (PROCESS_CSS + '<div class="rag-process"><div class="rag-empty">Ask a question: planning, every search '
                "with its retrieved chunks and original transcript text, and the checks appear here step by step."
                "</div></div>")
    cards = step_cards(steps)
    html = [PROCESS_CSS, '<div class="rag-process">']
    for number, (title, body, kind) in enumerate(cards, 1):
        html.append(f'<div class="rag-card {kind}"><div class="rag-head"><span class="rag-step">STEP {number}</span>'
                    f'<span>{escape(title)}</span></div><div class="rag-body">{body}</div></div>')
    if running:
        html.append(f'<div class="rag-card pending"><div class="rag-head"><span class="rag-step">NOW</span>'
                    f'<span>{escape(current_activity(steps))}</span></div></div>')
    elif outcome:
        status, text = outcome
        kind = "ok" if status == "answered" else "fail"
        title = "🏁 Finished — answer verified" if status == "answered" else f"🏁 Finished — {status.replace('_', ' ')}"
        html.append(f'<div class="rag-card {kind}"><div class="rag-head"><span class="rag-step">DONE</span>'
                    f'<span>{escape(title)}</span></div><div class="rag-body">{escape(text)}</div></div>')
    return "".join(html + ["</div>"])


def format_reply(status, answer, citations, metrics):
    """Turn one agent outcome into a chat message in Markdown."""

    if status == "answered":
        lines = [CITATION_ID.sub("", answer)]
        if citations:
            lines.append("\n**Citations** (✓ verified word for word against the transcript)")
            lines += [quote_block(cite["quote"]) + f"\n>\n> — ✓ *{meeting_label(cite['meeting_id'])}*\n"
                      for cite in citations]
        if metrics.get("generation_model"):
            via = " → ".join(metrics.get("routes_used") or []) or metrics["generation_model"]
            lines.append(f"<sub>Model: {via} · {metrics.get('elapsed_seconds', '?')}s</sub>")
        return "\n".join(lines)
    if status == "rate_limited":
        return "🚫 **Rate limit reached.** " + answer
    if status == "error":
        return "⚠️ **Error.** " + answer
    return f"⚠️ **No verified answer ({status}).** {answer}"


def usage_rows():
    """Rows for the rate-limit status table, newest activity first."""

    def stamp(value):
        return value.strftime("%Y-%m-%d %H:%M:%S") if value else ""

    return [[("🔴 " if row["state"] == "rate limited" else "🟢 " if row["state"] == "ok" else "🟠 ") + row["state"],
             row["kind"], row["model"], row["requests"], row["ok"], row["rate_limited"],
             stamp(row["last_time"]), stamp(row["last_rate_limit"])]
            for row in usage_summary() if row["provider"] == PROVIDER]


def key_markdown():
    """Configured-key badge; never shows the credential value."""

    ok = key_status().get(PROVIDER)
    return ("**NVIDIA_API_KEY** (from `.env` or environment; value never shown): "
            + ("✅ set" if ok else "❌ not set — add it to .env and restart"))


def build_ui():
    """Create the Gradio app lazily so controller tests do not require Gradio."""

    import gradio as gr

    def model_choices(rows, checks):

        return [(row["model"] + " — " + checks.get(row["model"], {}).get("status", "not tested"), row["model"])
                for row in rows]

    def backup_update(state, model_value, selected=()):

        checks = (state or {}).get("checks", {})
        choices = [name for name, check in sorted(checks.items())
                   if check.get("status") == PASSED and name != model_value]
        return gr.Dropdown(choices=choices, value=[name for name in (selected or []) if name in choices][:2])

    def registry_outputs(rows, state, message, value=None, level="info"):

        checks = state.get("checks", {})
        choices = model_choices(rows, checks)
        default = load_generation_settings().model
        ids = [row["model"] for row in rows]
        if not ids:
            # Catalog unavailable: still allow the configured default model.
            choices, ids = [(default + " — catalog unavailable", default)], [default]
        value = value if value in ids else (default if default in ids else ids[0])
        display = [dict(row, check=checks.get(row["model"], {})) for row in rows]
        note = alert_html([{"level": level, "title": "Model check", "text": message}]) if level != "info" else ""
        return (gr.Dropdown(choices=choices, value=value), f"*{message}*",
                {"message": message, "models": display}, state, note)

    def refresh_choices(state):

        from copy import deepcopy
        state = deepcopy(state or {"catalog": [], "checks": {}})
        level = "info"
        if not key_status().get(PROVIDER):
            rows, message, level = [], "NVIDIA_API_KEY is not set. Add it to .env (or Space secrets) and restart.", "warning"
        else:
            try:
                rows = discover_models(PROVIDER)
                message = (f"{len(rows)} NVIDIA models listed. Listed models are candidates; "
                           "test one to check live JSON generation.")
            except (ValueError, GenerationError, OSError) as error:
                rows, message, level = [], str(error), "warning"
        state["catalog"] = rows
        return registry_outputs(rows, state, message, level=level)

    def test_choice(selected_model, state, selected_backups):

        from copy import deepcopy
        state = deepcopy(state or {"catalog": [], "checks": {}})
        rows = state.get("catalog", [])
        level = "info"
        if not selected_model:
            message, level = "Select a model first.", "warning"
        else:
            try:
                checked = probe_model(PROVIDER, selected_model)
                state["checks"][selected_model] = checked
                message = checked["status"] + ". This checks small JSON output, not RAG accuracy or full-context capacity."
                if checked["status"] == "rate limited":
                    level = "error"
                    message = (f"{selected_model} is rate limited right now (HTTP 429). "
                               "Pick another model, or wait for the quota to reset.")
                elif checked["status"] != PASSED:
                    level = "warning"
                    message += " " + checked.get("reason", "")
            except (ValueError, GenerationError, OSError) as error:
                message, level = str(error), "warning"
        outputs = registry_outputs(rows, state, message, value=selected_model, level=level)
        return (*outputs, backup_update(state, selected_model, selected_backups), usage_rows())

    def respond(question, history, team, cutoff, method, limit, rounds, model_value,
                use_backups, selected_backups, state):

        history = list(history or [])
        question = (question or "").strip()
        if not question:
            yield history, "", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return
        history.append({"role": "user", "content": question})
        known = [row["model"] for row in (state or {}).get("catalog", [])]
        if not model_value or (known and model_value not in known and model_value != load_generation_settings().model):
            history.append({"role": "assistant",
                            "content": "⚠️ Select a model under **Model selection** (click Refresh models)."})
            yield history, "", process_html([]), "", "", {}, [], {}, {}, usage_rows()
            return
        backups = []  # Free-tier limits require a visible stop and manual model change.
        base = list(history)
        for status, answer, citations, activity, evidence, result, metrics, alert, steps in stream_question(
                question, team, cutoff, method, int(limit), int(rounds), model=model_value, backups=backups):
            done = bool(result) or status in ("error", "rate_limited")
            # Chat keeps only a one-line status, then the answer; full steps go to the process panel.
            reply = (format_reply(status, answer, citations, metrics) if done else
                     f"*{current_activity(steps)}*  \n<sub>Each step appears in **Agent process** below.</sub>")
            history = base + [{"role": "assistant", "content": reply}]
            panel = process_html(steps, running=not done,
                                 outcome=(status, CITATION_ID.sub("", answer or "")) if done else None)
            yield (history, "", panel, alert, activity, evidence, citations, result, metrics,
                   usage_rows() if done else gr.skip())

    teams = sorted({path.stem[:-1] for path in PARSED_DIR.glob('*.json')
                    if re.fullmatch(r'[A-Z]{2}\d{4}[a-d]', path.stem)})
    with gr.Blocks(title="Meeting RAG Assistant", fill_width=True,
                   analytics_enabled=False) as demo:
        gr.Markdown("# Meeting RAG Assistant\nAsk about the meeting transcripts. The agent searches, checks "
                    "evidence and verifies every quote before answering.")
        gr.Markdown(readiness())
        registry = gr.State({"catalog": [], "checks": {}})

        # ---------------- Chat ----------------
        alert = gr.HTML("", label="Rate-limit alerts")
        chatbot = gr.Chatbot(label="Chat", height=320, buttons=["copy"],
                             placeholder="Ask about any meeting, e.g. *Which teams chose a kinetic power "
                                         "source?* — or pick a team below to narrow the search.")
        with gr.Row():
            team = gr.Dropdown([("All teams", ALL_TEAMS)] + [(name, name) for name in teams], value=ALL_TEAMS,
                               label="Team", scale=1, filterable=True,
                               info="A team named in your question takes priority over All teams. Otherwise searches every team.")
            cutoff = gr.Dropdown([("All (a–d)", "all")] + [(letter, letter) for letter in "abcd"],
                                 value="all", label="Meetings", scale=1,
                                 info="A letter includes meetings up to that letter.")
            method = gr.Dropdown(["hybrid", "embedding", "bm25"], value="hybrid", label="Search method", scale=1,
                                 info="bm25 avoids NVIDIA embedding calls.")
        with gr.Row(equal_height=True):
            question = gr.Textbox(show_label=False, placeholder="Type your question and press Enter",
                                  lines=1, scale=8, autofocus=True)
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear chat", scale=1)
        gr.Markdown("<sub>Each question is answered independently; earlier chat turns are shown for reference "
                    "and are not sent to the model.</sub>")

        # ---------------- Live agent process (one step at a time) ----------------
        process = gr.HTML(process_html([]), label="Agent process — live", show_label=True, container=True,
                          max_height=640, autoscroll=True)

        # ---------------- Model selection ----------------
        with gr.Accordion("Model selection (NVIDIA)", open=True):
            gr.Markdown(key_markdown())
            model = gr.Dropdown([], label="Model",
                                info="Status after the dash: not tested / JSON check passed / rate limited")
            with gr.Row():
                refresh = gr.Button("Refresh models")
                test_model = gr.Button("Test selected model (one small request)")
            model_note = gr.Markdown("")
            with gr.Row():
                use_backups = gr.Checkbox(value=False, interactive=False,
                                          label="Automatic switching disabled — change models manually", scale=1)
                backups = gr.Dropdown([], value=[], multiselect=True, max_choices=2, scale=3,
                                      label="Backup NVIDIA models (up to 2, tried in order)",
                                      info="Only models that passed 'Test selected model' appear here.")
            with gr.Accordion("Rate-limit status (last 60 minutes)", open=True):
                usage = gr.Dataframe(value=usage_rows(), headers=USAGE_HEADERS, interactive=False, wrap=True)
                refresh_usage = gr.Button("Refresh status", size="sm")
            with gr.Accordion("Catalog details", open=False):
                catalog = gr.JSON(label="Catalog and live check results")
            gr.Markdown("<sub>Rate limits stop the run and appear in the agent box. Change models manually; "
                        "no automatic upgrade or switch is performed. Hybrid/embedding search uses NVIDIA embeddings.</sub>")

        with gr.Accordion("Search settings", open=False):
            limit = gr.Slider(1, 10, value=5, step=1, label="Chunks per search", precision=0)
            rounds = gr.Slider(1, 3, value=2, step=1, label="Maximum search rounds", precision=0)
        with gr.Accordion("Raw run data (JSON)", open=False):
            activity = gr.Textbox(label="Raw step log", lines=10, interactive=False)
            evidence = gr.JSON(label="Evidence used by the agent")
            citations = gr.JSON(label="Verified citations")
            metrics = gr.JSON(label="Duration, models used, API/cache counts")
            result = gr.JSON(label="Full result and trace")

        # ---------------- Events ----------------
        registry_outputs_list = [model, model_note, catalog, registry, alert]
        refresh.click(refresh_choices, [registry], registry_outputs_list, api_visibility="private").then(
            backup_update, [registry, model, backups], backups, api_visibility="private")
        demo.load(refresh_choices, [registry], registry_outputs_list, api_visibility="private")
        model.change(backup_update, [registry, model, backups], backups, api_visibility="private")
        test_model.click(test_choice, [model, registry, backups],
                         registry_outputs_list + [backups, usage], api_visibility="private")
        chat_inputs = [question, chatbot, team, cutoff, method, limit, rounds, model,
                       use_backups, backups, registry]
        chat_outputs = [chatbot, question, process, alert, activity, evidence, citations, result, metrics, usage]
        for trigger in (send.click, question.submit):
            trigger(respond, chat_inputs, chat_outputs, concurrency_limit=1, api_visibility="private")
        clear.click(lambda: ([], process_html([]), "", "", {}, [], {}, {}), None,
                    [chatbot, process, alert, activity, evidence, citations, result, metrics],
                    api_visibility="private")
        refresh_usage.click(usage_rows, None, usage, api_visibility="private")
        gr.Timer(30).tick(usage_rows, None, usage, api_visibility="private")
    return demo
