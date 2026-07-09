#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Fully local Pipecat Voice Agent (Apple Silicon / MLX)

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech,
running entirely on-device:

- Qwen3-ASR  (Speech-to-Text) via mlx-audio  -> services_local.Qwen3ASRSTTService
- LM Studio  (LLM)           OpenAI-compatible endpoint at http://localhost:1234/v1
- Qwen3-TTS  (Text-to-Speech) via mlx-audio  -> services_local.Qwen3TTSService

Requirements:
- macOS on Apple Silicon (arm64)
- LM Studio running locally with the model loaded and its server started

Run the bot using::

    uv run bot.py
"""

import asyncio
import collections
import json
import os
import re
import shutil
import time
from types import SimpleNamespace
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)


def _hf_offline_gate():
    """Enable HF offline mode when every model is already cached.

    MUST run before anything imports huggingface_hub — it freezes
    HF_HUB_OFFLINE at import time. Without this, a dead connection stalls
    startup ~10s per model file on hub freshness checks.
    """
    if os.getenv("HF_HUB_OFFLINE") is not None:
        return  # explicit user choice wins
    cache = os.getenv("HF_HUB_CACHE") or os.path.join(
        os.getenv("HF_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        "hub",
    )
    repos = [
        os.getenv("QWEN3_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16"),
        os.getenv("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
        "speechbrain/spkrec-ecapa-voxceleb",  # speaker-gate voiceprint encoder
    ]

    def cached(repo: str) -> bool:
        p = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots")
        return os.path.isdir(p) and bool(os.listdir(p))

    if all(cached(r) for r in repos):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        # Stay online to download, but fail fast on a dead network.
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "2")


_hf_offline_gate()

import aiohttp
from loguru import logger
from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIFunctionCallReportLevel,
    RTVIObserverParams,
    RTVIServerMessageFrame,
)
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
from pipecat.utils.string import TextPartForConcatenation
from pipecat.services.llm_service import FunctionCallParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

from mcp_toolsets import (
    TOOLSETS,
    call_mcp_tool,
    ensure_toolset_schemas,
    load_toolset_impl,
    proxy_handler,
    toolset_catalog,
)
from notes_memory_store import NotesMemoryStore
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from services_local import (
    GatedInterruptionVADTurnStartStrategy,
    Qwen3ASRSTTService,
    Qwen3TTSService,
    SpeakablePathFilter,
    SpeakableSymbolFilter,
    ThinkTagFilter,
    VoiceOnlyInterruptor,
    filler_wavs,
)

load_dotenv(override=True)

# Long-term memory lives in Apple Notes (folder = agent name). A SQLite
# sidecar holds ids + recall stats (+ future caches). The old memories.db was
# imported into Notes once (2026-07-08) and is now just an inert backup.
_SIDECAR_DB = os.path.join(os.path.dirname(__file__), "notes_memory.db")
memory = NotesMemoryStore(
    os.getenv("AGENT_NAME").strip(),
    _SIDECAR_DB,
    call_mcp_tool,
)

# Seeded action memories (kind='action') = TOOL QUIRKS / best-practice for the
# tools: objective, reusable facts about how a tool behaves and how to use it
# for reliable results — NOT Thomas' personal preferences (those are ordinary
# memories, kind='fact'). Seeded once at startup; edits by voice stick —
# "Agent, when searching mail, also ..." updates the note.
# Seed tool-quirk memories (kind 'action'). Recall matches on the trigger
# phrase BEFORE the first colon, and auto-injection truncates at ~220 chars —
# so each keeps a keyword-rich trigger and front-loads its most useful quirk.
SEED_ACTION_MEMORIES = [
    "Searching or finding emails or messages: search mailbox 'All' across "
    "accounts, since mail lives in Archive as well as the Inbox, and use a broad "
    "sender name over an exact address, bounding dates with date_from/date_to.",
    "Older or missing emails, only the newest showing: max_results caps before "
    "sorting, so reach older mail with date_from and date_to, a higher max_results, "
    "or offset, and broaden the filters when a search finds nothing.",
    "Deleting, trashing, or cleaning up emails: there is no delete tool, so use "
    "manage_trash with action 'move_to_trash', and set dry_run to false to actually "
    "move it, since it only previews otherwise.",
    "Reading one specific email in full: search and inbox show only a preview, so "
    "use read_email by sender and/or subject for the full body, and if it returns "
    "continue_offset, call it again with that offset until done.",
]


def _net_error(action: str, exc: Exception) -> dict:
    """Uniform error result for internet tools.

    Connection/DNS failures become an explicit offline report so the model
    can say plainly that it's offline, instead of relaying a raw exception.
    """
    import socket

    cause = exc
    while cause is not None:
        if isinstance(cause, (aiohttp.ClientConnectorError, ConnectionError, socket.gaierror)):
            return {
                "offline": True,
                "error": f"offline: {action} needs an internet connection and none is available right now",
            }
        cause = cause.__cause__ or getattr(cause, "os_error", None)
    if isinstance(exc, asyncio.TimeoutError):
        return {"error": f"{action} timed out — the network may be down or very slow"}
    return {"error": f"{action} failed: {exc}"}


async def _xai_responses(input_messages: list, tools: list, timeout: int = 60,
                         action: str = "web search") -> dict:
    """Call xAI's Responses API with server-side tools; parse answer+citations.

    Shared by the quick searches and the deeper escalate_to_grok tool.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return {"error": "XAI_API_KEY not configured"}

    payload = {
        "model": os.getenv("XAI_MODEL", "grok-4.5"),
        "input": input_messages,
        "tools": tools,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/responses",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                data = await response.json()
                if response.status != 200:
                    msg = data.get("error", data)
                    return {"error": f"xAI request failed ({response.status}): {msg}"}
    except Exception as exc:  # noqa: BLE001
        return _net_error(action, exc)

    # Collect the output text and any citations, tolerating shape variations.
    texts, citations = [], []
    for item in data.get("output", []):
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                texts.append(part.get("text", ""))
                for ann in part.get("annotations") or []:
                    url = ann.get("url") if isinstance(ann, dict) else None
                    if url:
                        citations.append(url)
    citations.extend(data.get("citations") or [])
    answer = "\n".join(t for t in texts if t).strip()
    if not answer:
        return {"error": "xAI returned no text"}
    return {"answer": answer, "citations": list(dict.fromkeys(citations))[:8]}


async def _xai_search(query: str, tool_type: str) -> dict:
    """Quick search through one xAI server-side tool ("web_search"/"x_search")."""
    return await _xai_responses(
        [{"role": "user", "content": query}], [{"type": tool_type}]
    )


async def escalate_to_grok(params: FunctionCallParams):
    """Tool handler: hand a hard query to Grok (xAI) for deep synthesis.

    Grok answers with live web + X search available; it has NO access to the
    user's files, memory, calendar, or mail — pass everything it needs in the
    query. Use when the question needs careful synthesis or the local model is
    not confident (high hallucination risk).
    """
    query = str(params.arguments.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "empty query"})
        return
    context_note = str(params.arguments.get("context") or "").strip()
    logger.info(f"escalate_to_grok: [{query[:80]}]")
    instructions = (
        "You are a careful expert consultant. Reason step by step and give a "
        "thorough, well-grounded synthesis. Prefer verified facts; use web and "
        "X search to check anything uncertain. If evidence is thin or "
        "conflicting, say so plainly rather than guessing. Cite key sources."
    )
    user = query if not context_note else f"{query}\n\nContext from the user:\n{context_note}"
    result = await _xai_responses(
        [{"role": "system", "content": instructions},
         {"role": "user", "content": user}],
        [{"type": "web_search"}, {"type": "x_search"}],
        timeout=120,
        action="the escalation to Grok",
    )
    if "error" in result and "not configured" in result["error"]:
        result["hint"] = "XAI_API_KEY is required for escalation"
    await params.result_callback(result)


async def _google_search(query: str) -> dict:
    """Fallback web search via Google Custom Search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                    "cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                    "q": query,
                    "num": 5,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                data = await response.json()
    except Exception as exc:  # noqa: BLE001
        return _net_error("web search", exc)
    if "error" in data:
        return {"error": data["error"].get("message", "search error")}
    results = [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
        }
        for item in data.get("items", [])[:5]
    ]
    return {"results": results or "no results found"}


async def google_search(params: FunctionCallParams):
    """Tool handler: fast Google search returning the top 5 results."""
    query = str(params.arguments.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "empty query"})
        return
    logger.info(f"google_search: [{query}]")
    await params.result_callback(await _google_search(query))


async def x_web_search(params: FunctionCallParams):
    """Tool handler: deep agentic web search via xAI (slower, reads pages)."""
    query = str(params.arguments.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "empty query"})
        return
    logger.info(f"x_web_search: [{query}]")
    result = await _xai_search(query, "web_search")
    if "error" in result and "not configured" in result["error"]:
        result["hint"] = "use google_search instead"
    await params.result_callback(result)


async def x_search(params: FunctionCallParams):
    """Tool handler: search posts and discussions on X (Twitter) via xAI."""
    query = str(params.arguments.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "empty query"})
        return
    logger.info(f"x_search: [{query}]")
    await params.result_callback(await _xai_search(query, "x_search"))


async def detect_lmstudio_model(base_url: str) -> str | None:
    """Return the id of the model currently loaded in LM Studio, if any.

    Uses LM Studio's REST API (/api/v0/models), which reports load state.
    """
    root = base_url.rsplit("/v1", 1)[0]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{root}/api/v0/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                data = await response.json()
        loaded = [
            m["id"]
            for m in data.get("data", [])
            if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm")
        ]
        return loaded[0] if loaded else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not query LM Studio for loaded models: {exc}")
        return None


async def detect_lmstudio_context_window(base_url: str, model_id: str | None) -> int | None:
    """The loaded model's configured context length, from the same REST API."""
    root = base_url.rsplit("/v1", 1)[0]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{root}/api/v0/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                data = await response.json()
        for m in data.get("data", []):
            if m.get("state") != "loaded" or m.get("type") not in ("llm", "vlm"):
                continue
            if model_id and m.get("id") != model_id:
                continue
            return m.get("loaded_context_length") or m.get("max_context_length")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not query LM Studio for the context window: {exc}")
    return None


def local_timezone_name() -> str:
    """Return the system's IANA timezone name (e.g. 'Asia/Singapore')."""
    try:
        # /etc/localtime -> /var/db/timezone/zoneinfo/Asia/Singapore (macOS)
        # or /usr/share/zoneinfo/Asia/Singapore (Linux)
        link = os.readlink("/etc/localtime")
        return link.split("zoneinfo/")[-1]
    except OSError:
        return datetime.now().astimezone().tzname() or "unknown"


async def get_current_time(params: FunctionCallParams):
    """Tool handler: report the current local date, time, and timezone."""
    now = datetime.now().astimezone()
    await params.result_callback(
        {
            "datetime": now.strftime("%A, %d %B %Y, %-I:%M:%S %p"),
            "timezone": f"{local_timezone_name()} (UTC{now.strftime('%z')})",
            "iso": now.isoformat(),
        }
    )


get_current_time_schema = FunctionSchema(
    name="get_current_time",
    description="Get the current local date, time, and timezone.",
    properties={},
    required=[],
    handler=get_current_time,
)


async def open_in_safari(params: FunctionCallParams):
    """Tool handler: open a URL in Safari on this Mac."""
    url = str(params.arguments.get("url", "")).strip()
    # Only allow web URLs — `open` would happily run file:// or other schemes.
    if not url.lower().startswith(("http://", "https://")):
        if url and "." in url and " " not in url:
            url = f"https://{url}"
        else:
            await params.result_callback({"error": f"not a valid web URL: {url!r}"})
            return

    logger.info(f"open_in_safari: [{url}]")
    proc = await asyncio.create_subprocess_exec(
        "open", "-a", "Safari", url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        await params.result_callback(
            {"error": f"could not open Safari: {stderr.decode(errors='ignore').strip()}"}
        )
        return
    await params.result_callback({"status": "opened", "url": url})


def _recent_conversation(context, max_messages: int = 6) -> str:
    """Snapshot the last few user/assistant turns as plain text."""
    lines = []
    for msg in context.get_messages():
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines[-max_messages:])


async def remember(params: FunctionCallParams):
    """Tool handler: store a new memory, or edit an existing one by id."""
    content = str(params.arguments.get("content", "")).strip()
    if not content:
        await params.result_callback({"error": "nothing to remember"})
        return
    memory_id = params.arguments.get("id")
    if memory_id is not None:
        memory_id = str(memory_id).strip() or None
    snapshot = _recent_conversation(params.context)
    person = str(params.arguments.get("person") or "").strip() or None
    kind = str(params.arguments.get("kind") or "").strip().lower() or None
    result = await asyncio.to_thread(
        memory.remember, content, snapshot, memory_id, person, kind
    )
    if "error" in result:
        await params.result_callback(result)
        return
    verb = "edited" if result.get("edited") else "remembered"
    logger.info(f"remember: [{content[:80]}] -> {verb} id {result['id']}")
    await params.result_callback({"status": verb, **result})


async def forget(params: FunctionCallParams):
    """Tool handler: delete a memory by id."""
    memory_id = str(params.arguments.get("id") or "").strip()
    if not memory_id:
        await params.result_callback({"error": "a memory id is required"})
        return
    result = await asyncio.to_thread(memory.forget, memory_id)
    logger.info(f"forget: id {memory_id} -> {result}")
    await params.result_callback(result)


# Live LLM context, set at pipeline start — lets recall skip memories the
# MemoryInjector already placed in the conversation.
_LIVE_CONTEXT: dict = {"context": None}


def _injected_memory_text() -> str:
    ctx = _LIVE_CONTEXT.get("context")
    if ctx is None:
        return ""
    return "\n".join(
        str(m.get("content", ""))
        for m in ctx.get_messages()
        if isinstance(m, dict)
        and str(m.get("content", "")).startswith(("Recent context", "Action notes"))
    )


async def recall(params: FunctionCallParams):
    """Tool handler: search memories by keywords and/or a person filter."""
    keywords = params.arguments.get("keywords") or []
    if isinstance(keywords, str):
        keywords = keywords.split()
    person = str(params.arguments.get("person") or "").strip() or None
    kind = str(params.arguments.get("kind") or "").strip().lower() or None
    result = await asyncio.to_thread(
        memory.recall, list(keywords), person, 5, kind
    )
    logger.info(f"recall: {keywords} person={person} -> {result.get('error') or len(result['memories'])}")
    if result.get("error"):
        await params.result_callback(result)
        return
    candidates = result.get("candidates")
    if candidates:
        names = [c["name"] for c in candidates]
        await params.result_callback({
            "candidates": names,
            "note": ("Several people match that name — recall again with one exact "
                     f"name in `person`: {', '.join(names)}."),
        })
        return
    memories = [
        {
            "id": m["id"],
            "content": m["content"],
            "person": m.get("person"),
            "date": m["created_at"][:10],
            "kind": m.get("kind", "fact"),
        }
        for m in result["memories"]
    ]
    # Precision: OR-matching can drag in barely-related entries — keep
    # only memories that share a stem with the query or belong to a matched
    # profile (falling back to the full set if that would leave nothing).
    matched = set(result.get("matched_people") or [])
    stems = {
        w[:4] for kw in keywords
        for w in re.findall(r"[\w']+", str(kw).lower()) if len(w) > 2
    }
    if stems or matched:
        def _relevant(m):
            if (m.get("person") or "").lower() in matched:
                return True
            hay = f"{m['content']} {m.get('person') or ''}".lower()
            return any(w[:4] in stems for w in re.findall(r"[\w']+", hay) if len(w) > 2)
        strict = [m for m in memories if _relevant(m)]
        if strict:
            memories = strict
    # Token diet: plain-text lines instead of JSON, and memories the injector
    # already placed in the conversation shrink to an id reference.
    injected = _injected_memory_text()
    lines, shown = [], []
    for m in memories:
        if injected and m["content"] in injected:
            shown.append(str(m["id"]))
            continue
        tags = m["person"] or ""
        marker = ", action" if m.get("kind") == "action" else ""
        head = f"[{m['id']}{marker}, {m['date']}" + (f", {tags}" if tags else "") + "]"
        lines.append(f"- {head} {m['content']}")
    if shown:
        lines.append(f"(ids {', '.join(shown)} already shown in context above)")
    out = {"memories": "\n".join(lines) or "no matching memories"}
    if result.get("person"):
        out["person"] = result["person"]
    await params.result_callback(out)


async def add_person(params: FunctionCallParams):
    """Tool handler: register a person or add aliases/identifiers to them."""
    name = str(params.arguments.get("name", "")).strip()
    aliases = params.arguments.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    result = await asyncio.to_thread(memory.add_person, name, list(aliases))
    logger.info(f"add_person: {name} aliases={aliases} -> {result}")
    await params.result_callback(result)


async def edit_person(params: FunctionCallParams):
    """Tool handler: correct or extend a registered person (rename included)."""
    person = str(params.arguments.get("person", "")).strip()
    new_name = (str(params.arguments.get("new_name") or "").strip()) or None
    aliases = params.arguments.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    result = await asyncio.to_thread(
        memory.edit_person, person, new_name, list(aliases)
    )
    logger.info(f"edit_person: {person} -> {result}")
    await params.result_callback(result)


async def list_people(params: FunctionCallParams):
    """Tool handler: list all registered people."""
    people = await asyncio.to_thread(memory.list_people)
    # Drop empty fields and trim timestamps to dates to save tokens.
    people = [
        {k: v for k, v in {
            "name": p["name"], "aliases": p["aliases"],
            "since": p["since"][:10], "memories": p["memories"],
        }.items() if v}
        for p in people
    ]
    await params.result_callback({"people": people or "no people registered yet"})


remember_schema = FunctionSchema(
    name="remember",
    description=(
        "Store ONE lean, precise fact in long-term memory: a single tight "
        "sentence, no conversational filler. Split unrelated facts into "
        "separate calls. Update an existing memory by passing its id instead "
        "of storing a near-duplicate (recall first to check)."
    ),
    properties={
        "content": {
            "type": "string",
            "description": "The fact or note to remember, phrased so it makes sense on its own",
        },
        "id": {
            "type": "string",
            "description": "Optional: id of an existing memory (from recall) to overwrite with the new content",
        },
        "kind": {
            "type": "string",
            "enum": ["fact", "action"],
            "description": (
                "fact (default) = anything true, INCLUDING how he likes tasks "
                "done (his preferences). action = a TOOL QUIRK: how a tool "
                "behaves or the best way to use it for reliable results — "
                "objective and reusable, not personal. Store a tool quirk you "
                "hit as action; store a preference of his as a fact."
            ),
        },
        "person": {
            "type": "string",
            "description": (
                "Optional: registered person this is about. Ambiguous references "
                "return candidates — ask which one is meant."
            ),
        },
    },
    required=["content"],
    handler=remember,
)


forget_schema = FunctionSchema(
    name="forget",
    description=(
        "Delete a memory from long-term memory by its id (find the id with recall "
        "first). Use when the user asks you to forget something or a memory is wrong."
    ),
    properties={
        "id": {"type": "string", "description": "The id of the memory to delete (from recall)"},
    },
    required=["id"],
    handler=forget,
)


recall_schema = FunctionSchema(
    name="recall",
    description=(
        "Search your memory. Quick and always most relevant."
    ),
    properties={
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Keywords, matched with OR — any one can hit, so pass a long, generous "
                "list (name, topic, synonyms). More keywords finds more, never less."
            ),
        },
        "kind": {
            "type": "string",
            "enum": ["fact", "action"],
            "description": "Optional: 'action' = how-to notes for tasks; 'fact' = knowledge",
        },
        "person": {
            "type": "string",
            "description": (
                "Optional: the subject to look up — the person the question is "
                "Matches everyone who shares the name and returns all their memories. "
                "Leave empty to search everyone."
            ),
        },
    },
    required=["keywords"],
    handler=recall,
)


add_person_schema = FunctionSchema(
    name="add_person",
    description=(
        "Register or update a person (aliases merge). Record nicknames, "
        "handles, and emails as aliases."
    ),
    properties={
        "name": {"type": "string", "description": "The person's canonical name"},
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Nicknames and online identifiers, e.g. ['@handle', 'name@example.com']",
        },
    },
    required=["name"],
    handler=add_person,
)


edit_person_schema = FunctionSchema(
    name="edit_person",
    description=(
        "Correct or extend a registered person: fix the name's spelling "
        "(new_name — their memories follow, the old spelling stays as an "
        "alias), or add aliases. Use for any 'no, it's spelled ...' "
        "correction."
    ),
    properties={
        "person": {"type": "string", "description": "Current name or alias of the person"},
        "new_name": {"type": "string", "description": "Corrected canonical name, if renaming"},
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional nicknames/handles/emails to add",
        },
    },
    required=["person"],
    handler=edit_person,
)


list_people_schema = FunctionSchema(
    name="list_people",
    description="List all registered people with their aliases and memory counts.",
    properties={},
    required=[],
    handler=list_people,
)


HOME = os.path.expanduser("~")


def _home_display(path: str) -> str:
    """Home-relative form (~/...) for any path shown to the model."""
    return "~" + path[len(HOME):] if path.startswith(HOME + os.sep) else path


def _resolve_user_path(raw: str) -> str:
    """Resolve a model-provided path, tolerating home-relative spellings.

    The model mangles paths from the ~/... display: it drops the tilde
    (/Desktop/x.pdf, Desktop/x.pdf) or rebuilds an absolute path with the
    WRONG username (/Users/username/... vs the real home). Anything
    outside HOME or missing is retried as home-relative; the retry only
    wins if it lands under HOME and actually exists.
    """
    raw = str(raw).strip()
    path = os.path.realpath(os.path.expanduser(raw))
    if not path.startswith(HOME + os.sep) or not os.path.exists(path):
        # Derive a home-relative tail: strip a leading ~, or a wrong
        # /Users/<user>/ or /home/<user>/ prefix, else a leading slash.
        tail = raw
        if tail.startswith("~"):
            tail = tail[1:]
        else:
            m = re.match(r"^/(?:Users|home)/[^/]+/(.*)$", tail)
            if m:
                tail = m.group(1)
        candidate = os.path.realpath(os.path.join(HOME, tail.lstrip("/")))
        if candidate.startswith(HOME + os.sep) and os.path.exists(candidate):
            return candidate
    return path

# Identity: the assistant's name and its user's name (see .env). The short
# forms are what gets spoken; the full forms appear in the system prompt,
# ASR vocabulary, and wake words.
AGENT_NAME = os.getenv("AGENT_NAME").strip()
AGENT_NAME_SHORT = os.getenv("AGENT_NAME_SHORT").strip()
USER_NAME = os.getenv("USER_NAME").strip()
USER_NAME_SHORT = os.getenv("USER_NAME_SHORT").strip()

# File types the agent is allowed to open/read (text-like documents, code, PDFs).
OPENABLE_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rtf", ".pdf",
    ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".xml",
    ".py", ".js", ".ts", ".html", ".css", ".sh", ".toml", ".ini", ".conf",
    ".tex", ".sql", ".env", ".l4",
}


# Dependency/cache trees never hold the user's own documents.
EXCLUDED_DIR_SEGMENTS = {
    "node_modules", "__pycache__", "site-packages", "venv",
    "Caches", "DerivedData", "nltk_data",
}

# Human content files for the recently-touched list in the system prompt:
# notes and documents a personal assistant might be asked about — not code.
# Kept within OPENABLE_EXTENSIONS so read_file/open_file accept every entry.
RECENT_FILE_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rtf", ".pdf", ".csv", ".tsv",
}

# Extensions that suggest a personal document rather than code — ranked above
# code hits regardless of recency.
DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".pages", ".rtf", ".txt", ".md", ".csv",
    ".xlsx", ".numbers", ".pptx", ".key", ".eml", ".emlx",
}


def _match_snippet(path: str, words: list[str]) -> str | None:
    """Line of a small plain-text file matching the most query words."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdf", ".rtf") or ext not in OPENABLE_EXTENSIONS:
        return None
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > 256_000:
            return None
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    patterns = [re.compile(rf"\b{re.escape(w)}", re.IGNORECASE) for w in words]
    best, best_hits = None, 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        hits = sum(1 for p in patterns if p.search(stripped))
        if hits > best_hits:
            best, best_hits = stripped, hits
            if hits == len(patterns):
                break
    return best[:160] if best else None


async def find_files(params: FunctionCallParams):
    """Tool handler: search home-directory file contents via Spotlight (mdfind).

    Content matches rank first; filename matches are a secondary signal
    (both > content-only > name-only, then recency).
    """
    query = str(params.arguments.get("query", "")).strip()
    # "|" separates alternatives; within one alternative all words must match.
    alternatives = [
        (alt, re.findall(r"\w+", alt))
        for alt in (a.strip() for a in query.split("|"))
        if re.findall(r"\w+", alt)
    ]
    if not alternatives:
        await params.result_callback({"error": "empty query"})
        return
    words = [w for _alt, ws in alternatives for w in ws]

    async def mdfind(args: list[str], limit: int = 200) -> list[str]:
        proc = await asyncio.create_subprocess_exec(
            "mdfind", "-onlyin", HOME, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        paths: list[str] = []

        async def read_paths():
            while len(paths) < limit:
                line = await proc.stdout.readline()
                if not line:
                    return
                if p := line.decode(errors="ignore").strip():
                    paths.append(p)

        try:
            await asyncio.wait_for(read_paths(), timeout=10)
        except asyncio.TimeoutError:
            pass
        finally:
            if proc.returncode is None:
                proc.kill()
        return paths

    logger.info(f"find_files: [{query}]")
    # Content matching only reaches file types Spotlight extracts text from;
    # the name query still surfaces files it can't read inside (.env, .l4).
    content_pred = " || ".join(
        "(" + " && ".join(f'kMDItemTextContent = "{w}*"cdw' for w in ws) + ")"
        for _alt, ws in alternatives
    )
    plain_alts = [alt.replace('"', "").replace("\\", "") for alt, _ws in alternatives]
    name_pred = " || ".join(f'kMDItemFSName = "*{alt}*"cd' for alt in plain_alts)
    content_paths, name_paths = await asyncio.gather(
        mdfind([content_pred]), mdfind([name_pred])
    )

    content_set, name_set = set(content_paths), set(name_paths)
    ranked: list[tuple[int, float, str]] = []
    for p in dict.fromkeys(content_paths + name_paths):
        # Skip app-support noise, dependency trees, and hidden directories.
        parts = p.split(os.sep)
        if (
            "/Library/" in p
            or any(part.startswith(".") for part in parts)
            or any(part in EXCLUDED_DIR_SEGMENTS for part in parts)
        ):
            continue
        try:
            mtime = os.stat(p).st_mtime
        except OSError:
            continue
        # Content beats name-only; document types beat code regardless of age.
        score = (
            (4 if p in content_set else 0)
            + (2 if p in name_set else 0)
            + (1 if os.path.splitext(p)[1].lower() in DOCUMENT_EXTENSIONS else 0)
        )
        ranked.append((score, mtime, p))
    ranked.sort(reverse=True)

    # Token diet: plain-text lines, home-relative paths, and full detail only
    # for the top hits — the tail is paths alone. The path lines are accepted
    # verbatim by open_file/read_file (both expand ~).
    lines = []
    for i, (_score, mtime, p) in enumerate(ranked[:12]):
        display = _home_display(p)
        if i < 6:
            date = datetime.fromtimestamp(mtime).astimezone().strftime("%Y-%m-%d")
            marker = ", folder" if os.path.isdir(p) else ""
            lines.append(f"{display} ({date}{marker})")
            if snippet := _match_snippet(p, words):
                lines.append(f"  > {snippet}")
        else:
            lines.append(display)
    await params.result_callback({"files": "\n".join(lines) or "no files found"})


async def open_file(params: FunctionCallParams):
    """Tool handler: open a text file or PDF in its default macOS app."""
    path = _resolve_user_path(params.arguments.get("path", ""))
    if not path.startswith(HOME + os.sep):
        await params.result_callback({"error": "can only open files inside the home directory"})
        return
    if not os.path.isfile(path):
        await params.result_callback({"error": f"file not found: {_home_display(path)}"})
        return
    ext = os.path.splitext(path)[1].lower()
    if ext not in OPENABLE_EXTENSIONS:
        await params.result_callback(
            {"error": f"only text documents and PDFs can be opened, not {ext or 'files without extension'}"}
        )
        return

    logger.info(f"open_file: [{path}]")
    proc = await asyncio.create_subprocess_exec(
        "open", path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        await params.result_callback(
            {"error": f"could not open: {stderr.decode(errors='ignore').strip()}"}
        )
        return
    await params.result_callback({"status": "opened", "path": _home_display(path)})


async def run_javascript(params: FunctionCallParams):
    """Tool handler: run a JavaScript snippet in a sandboxed Node process."""
    code = str(params.arguments.get("code", "")).strip()
    if not code:
        await params.result_callback({"error": "no code provided"})
        return
    node = shutil.which("node")
    if not node:
        await params.result_callback({"error": "node is not installed"})
        return

    logger.info(f"run_javascript: [{code[:100]}]")
    proc = await asyncio.create_subprocess_exec(
        node, "--permission", "-",  # permission model: no fs/child_process/workers
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(code.encode()), timeout=8
        )
    except asyncio.TimeoutError:
        proc.kill()
        await params.result_callback({"error": "execution timed out after 8 seconds"})
        return

    out = stdout.decode(errors="ignore").strip()
    err = stderr.decode(errors="ignore").strip()
    if proc.returncode != 0:
        # Surface just the error message, not the whole node stack trace.
        first = next((l for l in err.splitlines() if "Error" in l), err[:300])
        await params.result_callback({"error": first[:300]})
        return
    await params.result_callback({"output": out[:2000] or "(no output — use console.log)"})


# WMO weather interpretation codes -> spoken-friendly text
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


async def get_weather(params: FunctionCallParams):
    """Tool handler: current weather + 3-day forecast via Open-Meteo (free, no key)."""
    location = str(params.arguments.get("location", "")).strip()
    if not location:
        await params.result_callback({"error": "no location given"})
        return

    logger.info(f"get_weather: [{location}]")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                geo = await response.json()
            places = geo.get("results") or []
            if not places:
                await params.result_callback({"error": f"could not find a place called {location!r}"})
                return
            place = places[0]

            async with session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                               "precipitation,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                             "precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": 3,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                data = await response.json()
    except Exception as exc:  # noqa: BLE001
        await params.result_callback(_net_error("the weather lookup", exc))
        return

    cur = data.get("current", {})
    daily = data.get("daily", {})
    forecast = [
        {
            "date": daily["time"][i],
            "conditions": WMO_CODES.get(daily["weather_code"][i], "unknown"),
            "high_c": daily["temperature_2m_max"][i],
            "low_c": daily["temperature_2m_min"][i],
            "rain_chance_pct": daily["precipitation_probability_max"][i],
        }
        for i in range(len(daily.get("time", [])))
    ]
    await params.result_callback(
        {
            "place": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "current": {
                "conditions": WMO_CODES.get(cur.get("weather_code"), "unknown"),
                "temperature_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"),
            },
            "forecast": forecast,
        }
    )


get_weather_schema = FunctionSchema(
    name="get_weather",
    description="Current conditions and 3-day forecast for any place. Use for all weather questions.",
    properties={
        "location": {"type": "string", "description": "City or place name"},
    },
    required=["location"],
    handler=get_weather,
)


async def _alpha_vantage(session: aiohttp.ClientSession, **query) -> dict:
    """One Alpha Vantage API call. Returns the JSON payload or {"error": ...}."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY not configured"}
    try:
        async with session.get(
            "https://www.alphavantage.co/query",
            params={**query, "apikey": api_key},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            data = await response.json(content_type=None)
    except Exception as exc:  # noqa: BLE001
        return _net_error("the market data lookup", exc)
    # The API reports problems (bad symbol, rate limit) as 200s with these keys.
    for key in ("Error Message", "Note", "Information"):
        if key in data:
            return {"error": str(data[key])[:300]}
    return data


async def _resolve_symbol(session: aiohttp.ClientSession, query: str) -> dict | None:
    """Best ticker match for a free-text company name, or None."""
    data = await _alpha_vantage(session, function="SYMBOL_SEARCH", keywords=query)
    matches = data.get("bestMatches") or []
    return matches[0] if matches else None


async def get_financial_info(params: FunctionCallParams):
    """Tool handler: stock quotes, company fundamentals and FX/crypto rates."""
    kind = str(params.arguments.get("kind", "quote")).strip()
    symbol = str(params.arguments.get("symbol", "")).strip()
    logger.info(f"get_financial_info: [{kind}] [{symbol}]")

    async with aiohttp.ClientSession() as session:
        if kind == "exchange_rate":
            frm = str(params.arguments.get("from_currency", "")).strip().upper()
            to = str(params.arguments.get("to_currency", "USD")).strip().upper()
            if not frm:
                await params.result_callback({"error": "from_currency is required for exchange_rate"})
                return
            data = await _alpha_vantage(
                session, function="CURRENCY_EXCHANGE_RATE",
                from_currency=frm, to_currency=to,
            )
            if "error" in data:
                await params.result_callback(data)
                return
            rate = data.get("Realtime Currency Exchange Rate", {})
            await params.result_callback(
                {
                    "from": rate.get("2. From_Currency Name") or frm,
                    "to": rate.get("4. To_Currency Name") or to,
                    "rate": rate.get("5. Exchange Rate"),
                    "as_of_utc": rate.get("6. Last Refreshed"),
                }
            )
            return

        if not symbol:
            await params.result_callback({"error": "no symbol or company name given"})
            return

        async def fetch(sym: str) -> dict | None:
            """One quote/overview lookup: payload, {'error': ...}, or None if empty."""
            fn = "OVERVIEW" if kind == "overview" else "GLOBAL_QUOTE"
            data = await _alpha_vantage(session, function=fn, symbol=sym)
            if "error" in data:
                return data
            payload = data if kind == "overview" else data.get("Global Quote") or {}
            return payload if payload.get("Symbol") or payload.get("01. symbol") else None

        # Try the input as a ticker first; fall back to resolving a company name.
        sym, note = symbol.upper(), None
        result = await fetch(sym)
        if result is None:
            match = await _resolve_symbol(session, symbol)
            if not match:
                await params.result_callback({"error": f"no listed security found for {symbol!r}"})
                return
            sym = match.get("1. symbol", sym)
            note = f"interpreted {symbol!r} as {sym} ({match.get('2. name', '')})"
            result = await fetch(sym)
        if result is None:
            await params.result_callback({"error": f"no data available for {sym}"})
            return
        if "error" in result:
            await params.result_callback(result)
            return

    if kind == "overview":
        out = {
            "symbol": result.get("Symbol"),
            "name": result.get("Name"),
            "exchange": result.get("Exchange"),
            "currency": result.get("Currency"),
            "sector": result.get("Sector"),
            "industry": result.get("Industry"),
            "market_cap": result.get("MarketCapitalization"),
            "pe_ratio": result.get("PERatio"),
            "eps": result.get("EPS"),
            "dividend_yield": result.get("DividendYield"),
            "profit_margin": result.get("ProfitMargin"),
            "revenue_ttm": result.get("RevenueTTM"),
            "analyst_target_price": result.get("AnalystTargetPrice"),
            "week52_high": result.get("52WeekHigh"),
            "week52_low": result.get("52WeekLow"),
        }
        about = result.get("Description", "")
        if about:
            out["about"] = about[:300]
    else:
        out = {
            "symbol": result.get("01. symbol"),
            "price": result.get("05. price"),
            "change": result.get("09. change"),
            "change_pct": result.get("10. change percent"),
            "day_high": result.get("03. high"),
            "day_low": result.get("04. low"),
            "previous_close": result.get("08. previous close"),
            "volume": result.get("06. volume"),
            "as_of": result.get("07. latest trading day"),
        }
    if note:
        out["note"] = note
    await params.result_callback(out)


get_financial_info_schema = FunctionSchema(
    name="get_financial_info",
    description=(
        "Live market data: 'quote' = stock price and daily change; 'overview' = "
        "company fundamentals; 'exchange_rate' = currency/crypto rates. Prefer "
        "over web search for prices, rates, and financials."
    ),
    properties={
        "kind": {
            "type": "string",
            "enum": ["quote", "overview", "exchange_rate"],
            "description": "What to fetch",
        },
        "symbol": {
            "type": "string",
            "description": "Ticker or company name, e.g. 'AAPL' or 'Apple'. For quote and overview.",
        },
        "from_currency": {
            "type": "string",
            "description": "Currency or crypto code to convert from, e.g. 'EUR' or 'BTC'. For exchange_rate.",
        },
        "to_currency": {
            "type": "string",
            "description": "Currency code to convert to, defaults to 'USD'. For exchange_rate.",
        },
    },
    required=["kind"],
    handler=get_financial_info,
)


run_javascript_schema = FunctionSchema(
    name="run_javascript",
    description=(
        "Run a short JavaScript snippet in sandboxed Node.js and get its printed "
        "output. Use for ALL non-trivial math, dates, and conversions — compute "
        "here rather than in your head. No filesystem, network, or packages."
    ),
    properties={
        "code": {
            "type": "string",
            "description": "Code ending in console.log(...) of the result",
        },
    },
    required=["code"],
    handler=run_javascript,
)


# Caps chosen to keep file content from overwhelming the LLM context window:
# at most ~6000 chars (~1.5k tokens) enters the conversation per read_file call.
READ_MAX_CHARS = 6000        # max characters returned per call
EXCERPT_RADIUS = 400         # characters of context on each side of a keyword hit
MAX_EXCERPTS = 6


def _extract_text(path: str) -> tuple[str, list[int], int]:
    """Extract full text. Returns (text, page_start_offsets, page_count).

    page_start_offsets is empty for non-PDF files.
    """
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts, offsets, pos = [], [], 0
        for p in reader.pages:
            offsets.append(pos)
            t = (p.extract_text() or "") + "\n"
            parts.append(t)
            pos += len(t)
        return "".join(parts), offsets, len(reader.pages)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(), [], 0


def _page_of(offset: int, page_offsets: list[int]) -> int | None:
    if not page_offsets:
        return None
    page = 1
    for i, start in enumerate(page_offsets, start=1):
        if offset >= start:
            page = i
    return page


def _line_of(offset: int, line_starts: list[int]) -> int:
    """1-based line number containing a character offset."""
    import bisect

    return bisect.bisect_right(line_starts, offset)


OUTLINE_PATTERNS = {
    # code: top-level definitions
    "code": (r"^(def |class |function |const |export |func |fn |public |private )", 0),
    # markdown: headings
    "md": (r"^#{1,6} ", 0),
}


def _outline(path: str, text: str, lines: list[str], page_offsets: list[int],
             page_count: int, line_starts: list[int]) -> dict:
    """Structural map of the file for navigation, kept small."""
    import re as _re

    result: dict = {"total_lines": len(lines), "chars": len(text)}
    entries: list[str] = []
    ext = os.path.splitext(path)[1].lower()

    if page_count:
        result["pdf_pages"] = page_count
        for p in range(min(page_count, 40)):
            start = page_offsets[p]
            end = page_offsets[p + 1] if p + 1 < page_count else len(text)
            first = next((l.strip() for l in text[start:end].splitlines() if l.strip()), "")
            entries.append(f"page {p + 1}: {first[:70]}")
        if page_count > 40:
            entries.append(f"... {page_count - 40} more pages")
    elif ext in (".md", ".markdown"):
        for i, line in enumerate(lines, 1):
            if _re.match(r"^#{1,6} ", line):
                entries.append(f"L{i}: {line.strip()[:70]}")
    elif ext in (".py", ".js", ".ts", ".sh", ".swift", ".l4", ".sql"):
        for i, line in enumerate(lines, 1):
            if _re.match(r"^(async def |def |class |function |func |fn |const \w+ *=|export |CREATE |GIVEN |DECIDE )", line):
                entries.append(f"L{i}: {line.rstrip()[:70]}")
    if not entries:
        # generic: evenly sampled one-liners
        step = max(1, len(lines) // 20)
        for i in range(0, len(lines), step):
            if lines[i].strip():
                entries.append(f"L{i + 1}: {lines[i].strip()[:60]}")

    out, total = [], 0
    for e in entries:
        total += len(e) + 1
        if total > 2500:
            out.append(f"... {len(entries) - len(out)} more entries; read line ranges to explore")
            break
        out.append(e)
    result["outline"] = out
    result["hint"] = "use start_line/end_line to read a section"
    return result


def _read_file_sync(
    path: str,
    page: int | None,
    keywords: list[str],
    start_line: int | None,
    end_line: int | None,
    outline: bool,
) -> dict:
    """Blocking file read; runs in a thread. Returns content or error."""
    text, page_offsets, page_count = _extract_text(path)
    result: dict = {}
    if page_count:
        result["pdf_pages"] = page_count

    if not text.strip():
        return {**result, "warning": "no extractable text (scanned/image-only PDF?)"}

    lines = text.split("\n")
    line_starts = [0]
    for l in lines[:-1]:
        line_starts.append(line_starts[-1] + len(l) + 1)

    if outline:
        return _outline(path, text, lines, page_offsets, page_count, line_starts)

    # Keyword mode: merged excerpt windows around each match, with line anchors.
    if keywords:
        lowered = text.lower()
        spans = []
        for kw in keywords:
            kw = str(kw).strip().lower()
            start = 0
            while kw:
                i = lowered.find(kw, start)
                if i < 0:
                    break
                spans.append((max(0, i - EXCERPT_RADIUS), min(len(text), i + len(kw) + EXCERPT_RADIUS), kw))
                start = i + len(kw)
        if not spans:
            return {**result, "excerpts": [], "total_matches": 0,
                    "note": "no keyword matches; try other keywords, an outline, or line ranges"}
        spans.sort()
        merged = [list(spans[0][:2]) + [{spans[0][2]}]]
        for s, e, kw in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2].add(kw)
            else:
                merged.append([s, e, {kw}])
        excerpts, total = [], 0
        for s, e, kws in merged[:MAX_EXCERPTS]:
            snippet = text[s:e].strip()
            if total + len(snippet) > READ_MAX_CHARS:
                snippet = snippet[: READ_MAX_CHARS - total]
            item = {"matched": sorted(kws), "line": _line_of(s, line_starts), "text": snippet}
            pg = _page_of(s, page_offsets)
            if pg:
                item["page"] = pg
            excerpts.append(item)
            total += len(snippet)
            if total >= READ_MAX_CHARS:
                break
        result["excerpts"] = excerpts
        result["total_matches"] = len(spans)
        if len(merged) > len(excerpts):
            result["note"] = (
                f"{len(merged) - len(excerpts)} more matching segments not shown; "
                "narrow the keywords or read around a match with start_line"
            )
        result["hint"] = "expand any excerpt with start_line/end_line around its line"
        return result

    # Page mode (PDF): return one page.
    if page is not None and page_count:
        if not 1 <= page <= page_count:
            return {"error": f"page {page} out of range (1-{page_count})"}
        start = page_offsets[page - 1]
        end = page_offsets[page] if page < page_count else len(text)
        result["scope"] = f"page {page} of {page_count}"
        result["content"] = text[start:end].strip()[:READ_MAX_CHARS]
        return result

    # Line-range mode (any file type): precise, resumable navigation.
    if start_line is not None:
        n = len(lines)
        if start_line < 0:  # tail: -50 = last 50 lines
            start_line = max(1, n + start_line + 1)
        start_line = max(1, min(start_line, n))
        last = min(end_line, n) if end_line else n
        chunk, used = [], 0
        i = start_line
        while i <= last and used + len(lines[i - 1]) + 1 <= READ_MAX_CHARS:
            chunk.append(lines[i - 1])
            used += len(lines[i - 1]) + 1
            i += 1
        result["scope"] = f"lines {start_line}-{i - 1} of {n}"
        result["content"] = "\n".join(chunk)
        if i <= last:
            result["continue_from_line"] = i
        return result

    # Default: head of the document, cut at a line boundary, resumable.
    if len(text) <= READ_MAX_CHARS:
        result["content"] = text.strip()
        return result
    chunk, used = [], 0
    i = 0
    while i < len(lines) and used + len(lines[i]) + 1 <= READ_MAX_CHARS:
        chunk.append(lines[i])
        used += len(lines[i]) + 1
        i += 1
    result["scope"] = f"lines 1-{i} of {len(lines)}"
    result["content"] = "\n".join(chunk).strip()
    result["continue_from_line"] = i + 1
    result["truncated"] = (
        "continue with start_line, jump via outline=true, or search with keywords"
    )
    return result


async def read_file(params: FunctionCallParams):
    """Tool handler: read the contents of a text file or PDF."""
    path = _resolve_user_path(params.arguments.get("path", ""))
    page = params.arguments.get("page")
    keywords = params.arguments.get("keywords") or []
    if isinstance(keywords, str):
        keywords = keywords.split()
    if not path.startswith(HOME + os.sep):
        await params.result_callback({"error": "can only read files inside the home directory"})
        return
    if not os.path.isfile(path):
        await params.result_callback({"error": f"file not found: {_home_display(path)}"})
        return
    ext = os.path.splitext(path)[1].lower()
    if ext not in OPENABLE_EXTENSIONS:
        await params.result_callback({"error": f"only text documents and PDFs can be read, not {ext or 'files without extension'}"})
        return
    start_line = params.arguments.get("start_line")
    end_line = params.arguments.get("end_line")
    outline = bool(params.arguments.get("outline"))
    logger.info(
        f"read_file: [{path}] page={page} keywords={keywords} "
        f"lines={start_line}-{end_line} outline={outline}"
    )
    try:
        result = await asyncio.to_thread(
            _read_file_sync,
            path,
            int(page) if page else None,
            list(keywords),
            int(start_line) if start_line is not None else None,
            int(end_line) if end_line is not None else None,
            outline,
        )
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"could not read file: {exc}"}
    await params.result_callback(result)


read_file_schema = FunctionSchema(
    name="read_file",
    description=(
        "Read a document, PDF, or code file from the home directory. Output is "
        "capped per call — navigate to what you need: outline=true maps the structure "
        "with line numbers; keywords return excerpts around matches; "
        "start_line/end_line read a range (negative start = from the end); "
        "truncated responses include continue_from_line. Flow: outline or "
        "keywords first, then the relevant range. find_files locates files; "
        "open_file shows them on screen."
    ),
    properties={
        "path": {"type": "string", "description": "Absolute path of the file to read"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Return excerpts around these words, with line anchors",
        },
        "outline": {
            "type": "boolean",
            "description": "Return a structural map (line-numbered sections) instead of content",
        },
        "start_line": {
            "type": "integer",
            "description": "First line to read (1-based; negative = from the end, e.g. -50 for the tail)",
        },
        "end_line": {
            "type": "integer",
            "description": "Last line to read (defaults to as much as fits in the cap)",
        },
        "page": {"type": "integer", "description": "Specific PDF page number (1-based)"},
    },
    required=["path"],
    handler=read_file,
)


# Reading one specific email in full. The normal mail path can't: search/list
# only expose a 150-500 char preview, and the MCP proxy caps every result at
# 6000 chars. read_email fetches the WHOLE body server-side (search_emails with
# max_content_length=0, which call_mcp_tool returns uncapped) and then pages it
# out in READ_MAX_CHARS chunks, mirroring read_file's continue_from pagination.
_email_read_cache: dict[str, dict] = {}   # locator key -> {body, subject, sender, date, mailbox, others}


def _email_locator_key(sender: str, subject: str, mailbox: str) -> str:
    return "\x00".join((sender.strip().lower(), subject.strip().lower(), mailbox.strip().lower()))


async def read_email(params: FunctionCallParams):
    """Tool handler: return the full text of one specific email, paged."""
    a = params.arguments
    sender = (a.get("sender") or "").strip()
    subject = (a.get("subject") or "").strip()
    mailbox = (a.get("mailbox") or "All").strip() or "All"
    try:
        offset = max(0, int(a.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    if not sender and not subject:
        await params.result_callback(
            {"error": "give a sender and/or subject to identify which email to read"}
        )
        return
    logger.info(
        f"read_email: sender=[{sender}] subject=[{subject}] mailbox=[{mailbox}] offset={offset}"
    )

    key = _email_locator_key(sender, subject, mailbox)
    # A follow-up page reuses the body fetched on the first call — no second
    # trip through Mail — as long as the locator is unchanged.
    cached = _email_read_cache.get(key) if offset > 0 else None
    if cached is None:
        search_args = {
            "mailbox": mailbox,
            "include_content": True,
            "max_content_length": 0,        # 0 = full body (uncapped via call_mcp_tool)
            "output_format": "json",
            "limit": 5,
            "sort": "date_desc",
        }
        if sender:
            search_args["sender"] = sender
        if subject:
            search_args["subject_keyword"] = subject
        try:
            raw = await call_mcp_tool("apple-mail", "search_emails", search_args)
        except Exception as exc:  # noqa: BLE001
            await params.result_callback({"error": f"could not read email: {exc}"})
            return
        try:
            items = (json.loads(raw) or {}).get("items", [])
        except (ValueError, TypeError):
            items = []
        items = [it for it in items if (it.get("content_preview") or "").strip()]
        if not items:
            await params.result_callback(
                {"error": "no matching email with readable content found — "
                          "adjust sender/subject, or set mailbox to 'All'"}
            )
            return
        chosen = items[0]  # most recent match (sort=date_desc)
        cached = {
            "body": (chosen.get("content_preview") or "").strip(),
            "subject": chosen.get("subject", ""),
            "sender": chosen.get("sender", ""),
            "date": chosen.get("received_date", ""),
            "mailbox": chosen.get("mailbox", mailbox),
            "others": max(0, len(items) - 1),
        }
        _email_read_cache.clear()   # only the currently-open email is ever cached
        _email_read_cache[key] = cached

    body = cached["body"]
    total = len(body)
    if offset >= total:
        await params.result_callback(
            {"error": f"offset {offset} is past the end of this email ({total} chars)"}
        )
        return
    chunk = body[offset:offset + READ_MAX_CHARS]
    end = offset + len(chunk)
    result = {
        "subject": cached["subject"],
        "from": cached["sender"],
        "date": cached["date"],
        "mailbox": cached["mailbox"],
        "content": chunk,
        "chars": f"{offset}–{end} of {total}",
    }
    if end < total:
        result["continue_offset"] = end
        result["note"] = (
            f"{total - end} more characters remain — call read_email again with the "
            f"same sender/subject and offset={end} to read the next part."
        )
    if offset == 0 and cached.get("others"):
        result["also_matched"] = (
            f"{cached['others']} other email(s) matched; showing the most recent. "
            "Narrow by subject/sender/date to pick a different one."
        )
    await params.result_callback(result)


read_email_schema = FunctionSchema(
    name="read_email",
    description=(
        "Read the FULL text of one specific email. Use this whenever you need an "
        "email's complete body — searching or listing mail only shows a short "
        "preview that cuts long emails off. Identify the email by sender and/or "
        "subject (mailbox defaults to 'All'). Long emails are paged: if the "
        "response has continue_offset, call read_email again with the same "
        "sender/subject and offset=continue_offset to get the next part."
    ),
    properties={
        "sender": {"type": "string", "description": "Sender email or name identifying the email"},
        "subject": {"type": "string", "description": "Subject keyword identifying the email"},
        "mailbox": {
            "type": "string",
            "description": "Where to look: 'All' (default), 'INBOX', 'Archive', or a folder name",
        },
        "offset": {
            "type": "integer",
            "description": "Character offset to resume from (the continue_offset from a "
                           "previous read_email); omit to start at the beginning",
        },
    },
    required=[],
    handler=read_email,
)


find_files_schema = FunctionSchema(
    name="find_files",
    description=(
        "Search the user's home directory for files by words in their content; "
        "filename matches rank too (uses Spotlight). Returns paths with "
        "modification dates and, where possible, a matching line from the file."
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "Content keywords or a filename fragment. Space-separated words "
                "must all appear in the same file; separate alternative "
                "searches with | (e.g. 'rental|tenancy')"
            ),
        },
    },
    required=["query"],
    handler=find_files,
)


open_file_schema = FunctionSchema(
    name="open_file",
    description=(
        "Open a text document or PDF from the user's home directory on their "
        "screen, in its default app. Use find_files first if you only know a "
        "name or topic."
    ),
    properties={
        "path": {"type": "string", "description": "Absolute path of the file to open"},
    },
    required=["path"],
    handler=open_file,
)


open_in_safari_schema = FunctionSchema(
    name="open_in_safari",
    description=(
        "Open a web page in the Safari browser on the user's computer. Use when "
        "the user asks to open, show, or bring up a website or a search result."
    ),
    properties={
        "url": {"type": "string", "description": "The full web URL to open"},
    },
    required=["url"],
    handler=open_in_safari,
)


google_search_schema = FunctionSchema(
    name="google_search",
    description=(
        "Fast web search returning the top 5 Google results (title, snippet, "
        "url). Responds quickly — good default for simple facts, lookups, and "
        "anything a snippet can answer."
    ),
    properties={
        "query": {"type": "string", "description": "The search query"},
    },
    required=["query"],
    handler=google_search,
)


x_web_search_schema = FunctionSchema(
    name="x_web_search",
    description=(
        "Deep web search that browses and reads pages, returning a researched "
        "answer with citations. Use for questions needing depth or synthesis, not quick facts."
    ),
    properties={
        "query": {"type": "string", "description": "The research question"},
    },
    required=["query"],
    handler=x_web_search,
)


x_search_schema = FunctionSchema(
    name="x_search",
    description=(
        "Search posts and discussions on X (formerly Twitter). Use for opinions, "
        "reactions, trending topics, breaking news chatter, sport results, or what specific "
        "people are saying."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "What to search for on X, a topic, event, or person",
        },
    },
    required=["query"],
    handler=x_search,
)


escalate_to_grok_schema = FunctionSchema(
    name="escalate_to_grok",
    description=(
        "Escalate a hard question to Grok AI for deep "
        "synthesis or fact-critical answers or when you are not confident enough. "
        "Grok has live web and X search but NO access to other tools — include all non-public details in the "
        "query but mask private and identifiable information. "
        "Tell the user you're checking with Grok before you do. "
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "The full question for Grok, self-contained (Grok can't see this "
                "chat). Mask the user's private details with placeholders first."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Optional: facts from the conversation Grok needs but couldn't "
                "find online — with private identifiers masked by placeholders"
            ),
        },
    },
    required=["query"],
    handler=escalate_to_grok,
)


class _ParamsWithCallback:
    """Proxy for FunctionCallParams with a swapped result_callback."""

    def __init__(self, inner, callback):
        self._inner = inner
        self.result_callback = callback

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _with_current_time(handler):
    """Stamp the current local time into every tool result.

    The model then always has fresh time context (the system prompt only has
    the session start), without spending a turn on get_current_time.
    """

    async def wrapped(params):
        original = params.result_callback

        async def callback(result, **kwargs):
            if isinstance(result, dict):
                # Just HH:MM — date and timezone are in the system prompt.
                result = {
                    **result,
                    "current_time": datetime.now().astimezone().strftime("%-I:%M %p"),
                }
            await original(result, **kwargs)

        await handler(_ParamsWithCallback(params, callback))

    return wrapped


# Apply to all native tools (MCP tools get the same stamp in their proxy —
# see mcp_toolsets._proxy; get_current_time would be redundant).
for _schema in (
    google_search_schema, x_web_search_schema, x_search_schema,
    escalate_to_grok_schema,
    open_in_safari_schema, find_files_schema, open_file_schema,
    read_file_schema, read_email_schema, run_javascript_schema, get_weather_schema,
    remember_schema, recall_schema, forget_schema,
    add_person_schema, edit_person_schema, list_people_schema,
):
    _schema._handler = _with_current_time(_schema.handler)  # noqa: SLF001 — handler property has no setter

# Tool calls never cancel on generic interruptions: mid-speech barge-ins are
# raw VAD (speaker unknown), so noise or other voices must not kill in-flight
# work. The enrolled speaker's cancellation happens explicitly via the
# verified-speech hook in run_bot once the speaker gate confirms the voice.
for _schema in (
    google_search_schema, x_web_search_schema, x_search_schema,
    escalate_to_grok_schema,
    get_current_time_schema, open_in_safari_schema, find_files_schema,
    open_file_schema, read_file_schema, read_email_schema, run_javascript_schema,
    get_weather_schema, get_financial_info_schema, remember_schema,
    recall_schema, forget_schema, add_person_schema, edit_person_schema,
    list_people_schema,
):
    _schema._handler = tool_options(cancel_on_interruption=False)(_schema.handler)  # noqa: SLF001


# Every native tool, for sessions and for /api/chat (load_toolset is
# session-bound and excluded — the API preloads all MCP toolsets instead).
NATIVE_TOOL_SCHEMAS = (
    google_search_schema, x_web_search_schema, x_search_schema,
    escalate_to_grok_schema,
    get_current_time_schema, open_in_safari_schema, find_files_schema,
    open_file_schema, read_file_schema, read_email_schema, run_javascript_schema,
    get_weather_schema, get_financial_info_schema, remember_schema,
    recall_schema, forget_schema, add_person_schema, edit_person_schema,
    list_people_schema,
)


# Conversational filler that makes useless memory-search keywords.
_INJECT_STOPWORDS = {
    "that", "this", "with", "have", "what", "when", "where", "which", "your",
    "yeah", "okay", "thanks", "thank", "really", "about", "there", "then",
    "some", "well", "just", "like", "know", "right", "going", "want", "could",
    "would", "should", "please", "tell", "give", "make",
    "actually", "maybe", "little", "gonna", "kind", "sort", "stuff",
    "thing", "things", "doing", "does", "been", "will", "from", "they", "them",
    AGENT_NAME.lower(), AGENT_NAME_SHORT.lower(),
}


# Spoken before slow tool calls; pre-synthesized into the TTS phrase cache at
# startup so they play instantly even while the LLM saturates the GPU.
FILLER_LINES = ["Give me a moment", "Just a second", "Hang in there a sec", "One moment", "Orbiting on that", "On it"]

# Spoken when the context hits the token limit and gets compressed into a
# summary (a slow, full-reprefill operation). Primed like the fillers, but
# NOT part of the client's filler rotation.
REFOCUS_LINE = "I need a minute to refocus my energy. I'll be right back."


def _speech_transform(text_filter):
    """Adapt a BaseTextFilter to the TTS text-transform interface."""

    async def transform(text: str, aggregation_type: str) -> str:
        await text_filter.reset_interruption()
        return await text_filter.filter(text)

    return transform


class LMStudioLLMService(OpenAILLMService):
    """OpenAI-compatible service pointed at LM Studio.

    Qwen chat templates don't understand OpenAI's 'developer' role. Async
    tool results (cancel_on_interruption=False tools) arrive as developer
    messages — without this flag the template drops them and the model never
    sees completed results, answering "let me check" forever. Declaring no
    developer-role support makes pipecat's adapter convert them for us.

    It also records each completion's text VERBATIM as it streams. The
    assistant aggregator normally rebuilds the context message from the
    TTS-side sentence stream (blank lines become spaces, etc.); the exact
    copy lets run_bot store what the model actually generated, which keeps
    the next prompt a byte-exact continuation — required for LM Studio's
    cache to survive a turn on hybrid models like qwen3.5-122b-a10b.
    """

    supports_developer_role = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._verbatim_parts: list[str] = []
        self._verbatim_consumed = True

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMFullResponseStartFrame):
            self._verbatim_parts = []
            self._verbatim_consumed = False
        elif isinstance(frame, LLMTextFrame) and not self._verbatim_consumed:
            self._verbatim_parts.append(frame.text)
        await super().push_frame(frame, direction)

    def has_verbatim(self) -> bool:
        """True while the current completion's text has not been stored yet."""
        return not self._verbatim_consumed and bool("".join(self._verbatim_parts))

    def take_verbatim(self) -> str:
        """The current completion's exact text; consumed once per completion."""
        if self._verbatim_consumed:
            return ""
        self._verbatim_consumed = True
        return "".join(self._verbatim_parts)


def _parse_phrase_list(env_name: str, default: list[str]) -> list[str]:
    """Command phrases from .env — a JSON array or a comma-separated string."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        import json as _json
        v = _json.loads(raw)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except ValueError:
        pass
    return [p.strip() for p in raw.split(",") if p.strip()]


NEW_SESSION_PHRASES = _parse_phrase_list(
    "COMMAND_NEW_SESSION_PHRASES", ["start a new session", "start new session"]
)
MUTE_PHRASES = _parse_phrase_list(
    "COMMAND_MUTE_PHRASES",
    ["shut up", "mute yourself", "be quiet", "stop", "stop talking",
     "i'm not talking to you", "i am not talking to you", "that's it", "got it", "okay thanks"],
)


def _normalize_command(text: str, wake_words: list[str]) -> str:
    """Lowercase, strip punctuation, drop a leading wake word and please/now."""
    t = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    t = re.sub(r"\s+", " ", t).strip()
    for w in (wake_words):
        if t.startswith(w + " "):
            t = t[len(w) + 1:].strip()
            break
    t = re.sub(r"^please\s+", "", t)
    t = re.sub(r"\s+(please|now|thanks)$", "", t)
    return t


def classify_command(text: str, wake_words: list[str]) -> str | None:
    """Return 'new-session', 'mute', or None for a transcript/typed message."""
    t = _normalize_command(text, wake_words)
    norm = lambda ps: {_normalize_command(p, wake_words) for p in ps}
    if t in norm(NEW_SESSION_PHRASES):
        return "new-session"
    if t in norm(MUTE_PHRASES):
        return "mute"
    return None


class VoiceCommandInterceptor(FrameProcessor):
    """Catch spoken control phrases BEFORE they reach the LLM.

    Sits right after the STT service. A transcript that matches a configured
    command (new session / mute) is SWALLOWED — never forwarded to memory
    injection or the LLM — and the action is taken: the client is told to
    reload / show muted, and mute also re-arms the server wake gate.
    """

    def __init__(self, stt, **kwargs):
        super().__init__(**kwargs)
        self._stt = stt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            action = classify_command(frame.text, self._stt._wake_words)  # noqa: SLF001
            if action:
                logger.info(f"Voice command intercepted: {action} [{frame.text[:40]}]")
                if action == "mute":
                    self._stt.require_wake_word()
                await self.push_frame(
                    RTVIServerMessageFrame(data={"event": "command", "action": action})
                )
                return  # swallow — the LLM never sees it
        await self.push_frame(frame, direction)


class MemoryInjector(FrameProcessor):
    """Auto-recall: for every user utterance, search long-term memory with the
    utterance's own words and prepend the top matches to the LLM context.

    This makes memory independent of the model's tool discipline — the facts
    are in context even if the model never calls recall. Sits between STT and
    the user aggregator, so only utterances that passed the speaker and wake
    gates trigger a lookup.

    Each new injection removes the previous one from the context (via the
    shared LLMContext), so long sessions carry one memory block, not one per
    utterance. With MEMORY_INJECT_APPEND_ONLY=1 previous blocks stay put and
    only not-yet-injected memories are appended — rewriting history invalidates
    LM Studio's KV-cache prefix, so append-only trades a few context tokens for
    cheap prefill on every round.
    """

    MARK = "Recent context"
    ACTION_MARK = "Action notes"

    def __init__(self, context=None, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._append_only = (
            os.getenv("MEMORY_INJECT_APPEND_ONLY", "0").strip().lower()
            in ("1", "true", "yes")
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            try:
                await self._inject_for(frame.text)
            except Exception as exc:  # noqa: BLE001 — never block the utterance
                logger.warning(f"Memory injection failed: {exc}")
        await self.push_frame(frame, direction)

    async def _inject_for(self, text: str):
        words = [
            w for w in re.findall(r"[\w']+", text.lower())
            if len(w) > 3 and w not in _INJECT_STOPWORDS
        ][:10]
        if len(words) < 2:
            return  # trivial utterance ("thanks", "okay") — nothing to look up
        await self._inject_kind(
            words, text, "fact", self.MARK,
            "partial keyword preview, NOT the complete list — call recall for any "
            "name, date, or detail it lacks", 4,
        )
        await self._inject_kind(
            words, text, "action", self.ACTION_MARK,
            "how the user wants this done — follow these", 2,
        )

    async def _inject_kind(self, words, text, kind, mark, hint, limit):
        result = await asyncio.to_thread(memory.recall, words, None, limit, kind)
        memories = result.get("memories")
        if not isinstance(memories, list) or not memories:
            return
        # Only keep memories that genuinely share a stem with the utterance —
        # OR-matching alone drags in barely-related entries.
        stems = {w[:4] for w in words}
        def relevant(m):
            if kind == "action":
                # Actions match on their TRIGGER phrase only (the part before
                # the first colon) — matching the procedure body causes false
                # hits like "passed" (time) ~ "pass mailbox 'All'".
                hay = m["content"].split(":", 1)[0].lower()
            else:
                hay = f"{m['content']} {m.get('person') or ''}".lower()
            return any(w[:4] in stems for w in re.findall(r"[\w']+", hay) if len(w) > 3)
        memories = [m for m in memories if relevant(m)][:4]
        if self._context is not None:
            # Skip memories already in context: earlier injection blocks AND
            # the system prompt (recent/top-recalled sections) — the model
            # doesn't need the same memory twice in one request.
            existing = "\n".join(
                str(m.get("content", ""))
                for m in self._context.get_messages()
                if isinstance(m, dict)
                and (m.get("role") == "system"
                     or str(m.get("content", "")).startswith(mark))
            )
            memories = [m for m in memories if m["content"][:100] not in existing]
        if not memories:
            return
        lines = "\n".join(
            f"- [{m['id']}, {m['created_at'][:10]}]"
            f"{' (' + m['person'] + ')' if m.get('person') else ''} "
            f"{m['content'][:220] + '…' if len(m['content']) > 220 else m['content']}"
            for m in memories
        )
        logger.info(f"Memory injection ({kind}): {len(memories)} for [{text[:60]}]")
        # Purely visual: let the client flash a spark orb for the lookup.
        await self.push_frame(
            RTVIServerMessageFrame(data={"event": "memory-lookup", "count": len(memories)})
        )
        # Drop the previous injection so only one memory block rides along —
        # unless append-only (cache-friendly) mode keeps history immutable.
        if self._context is not None and not self._append_only:
            msgs = self._context.get_messages()
            kept = [
                m for m in msgs
                if not (isinstance(m, dict)
                        and str(m.get("content", "")).startswith(mark))
            ]
            if len(kept) != len(msgs):
                self._context.set_messages(kept)
        # Lands in the context just before the user message that triggered it.
        # USER role, not system: LM Studio reprocesses the whole prompt when a
        # system message appears mid-conversation (measured: ~9s vs 0.2s on
        # qwen3.5-122b-a10b), while user/assistant/tool blocks continue the
        # prompt cache cleanly. User role also keeps provenance unambiguous.
        await self.push_frame(
            LLMMessagesAppendFrame(
                messages=[{
                    "role": "user",
                    "content": f"{mark} ({hint}):\n{lines}",
                }],
                run_llm=False,
            )
        )


class AudioToLLMAttach(FrameProcessor):
    """Hand gated utterance AUDIO to an audio-native LLM (LLM_AUDIO_INPUT=1).

    Sits between the user aggregator and the LLM. When the STT stashed audio
    for the utterance that just became the newest user message, the message
    content is rewritten to [input_audio, transcript-note] parts. The
    transcript note keeps audio-blind servers working (they simply read the
    ASR text, i.e. today's behavior) and keeps the trail searchable.
    """

    def __init__(self, stt, context, **kwargs):
        super().__init__(**kwargs)
        self._stt = stt
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                self._attach_audio()
            except Exception as exc:  # noqa: BLE001 — never block the turn
                logger.warning(f"Audio attach failed: {exc}")
        await self.push_frame(frame, direction)

    def _attach_audio(self):
        audio = self._stt.take_llm_audio()
        if not audio:
            return
        for m in reversed(self._context.get_messages()):
            if not (isinstance(m, dict) and m.get("role") == "user"):
                continue
            if m.get("content") == audio["text"]:
                m["content"] = [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio["b64"], "format": "wav"},
                    },
                    {
                        "type": "text",
                        "text": f"[voice message — ASR transcript, may contain "
                                f"mishearings: {audio['text']}]",
                    },
                ]
                logger.info("Attached utterance audio to the LLM turn")
            break  # only ever consider the newest user message


def _recent_memories_block() -> str:
    """The five newest memories for the system prompt, so the model has
    recency awareness without a recall call."""
    try:
        recent = memory.recent(5, kind="fact")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not load recent memories: {exc}")
        return ""
    if not recent:
        return ""

    def _line(m):
        tags = m["person"] or ""
        date = m["created_at"][:10]
        marker = ", action" if m.get("kind") == "action" else ""
        return f"- [{m['id']}{marker}, {date}{', ' + tags if tags else ''}] {m['content']}"

    out = (
        "\nYour five most recent memories, newest first (recall has the rest):\n"
        + "\n".join(_line(m) for m in recent)
    )
    try:
        top = memory.top_recalled(5, exclude_ids={m["id"] for m in recent})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not load top-recalled memories: {exc}")
        top = []
    if top:
        out += (
            "\nYour most frequently recalled memories:\n"
            + "\n".join(_line(m) for m in top)
        )
    return out


async def _calendar_outlook_block() -> str:
    """The last 2 and next 6 calendar events for the system prompt.

    Fetched straight from the calendar MCP at session start, so the model
    has schedule awareness before the first exchange. Only the compact
    header lines survive — the tool's full notes/URLs stay out of the prompt.
    """
    now = datetime.now().astimezone()
    try:
        text = await asyncio.wait_for(
            call_mcp_tool(
                "calendar",
                "get_events",
                {
                    "start_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end_date": (now + timedelta(days=21)).strftime("%Y-%m-%d"),
                },
            ),
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 — a calendar hiccup must not block startup
        logger.warning(f"Could not load calendar for the system prompt: {exc}")
        return ""

    events: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Title: "):
            cur = {"title": line[7:]}
            events.append(cur)
        elif cur is None:
            continue
        elif line.startswith("Start: "):
            cur["start"] = line[7:]
        elif line.startswith("End: "):
            cur["end"] = line[5:]
        elif line.startswith("Location: "):
            cur["location"] = line[10:]
        elif line == "All-day event":
            cur["all_day"] = True

    def _dt(raw: str) -> datetime | None:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    parsed = sorted(
        ((d, e) for e in events if (d := _dt(e.get("start", ""))) is not None),
        key=lambda p: p[0],
    )
    local_now = now.replace(tzinfo=None)

    def _still_running(e: dict, start: datetime) -> bool:
        end = _dt(e.get("end", ""))
        if end is None:
            return False
        if e.get("all_day") and len(e.get("end", "")) == 10:
            end += timedelta(days=1)  # date-only end is inclusive
        return start <= local_now < end

    # "Last": anything that already started (over or still running) counts.
    past = [(d, e) for d, e in parsed if d <= local_now][-2:]
    upcoming = [(d, e) for d, e in parsed if d > local_now][:6]
    if not past and not upcoming:
        return ""

    def _fmt(d: datetime, e: dict, status: str = "") -> str:
        when = f"{d:%Y-%m-%d} all-day" if e.get("all_day") else f"{d:%Y-%m-%d %-I:%M %p}"
        if status:
            when += f" ({status})"
        loc = f" @ {e['location']}" if e.get("location") else ""
        return f"- {when}: {e['title']}{loc}"

    lines = ["\nCalendar events around now (get_events has the details):"]
    lines.extend(
        _fmt(d, e, "ongoing" if _still_running(e, d) else "already passed")
        for d, e in past
    )
    lines.extend(_fmt(d, e) for d, e in upcoming)
    return "\n" + "\n".join(lines)


async def _recent_files_block() -> str:
    """The 10 most recently modified supported files in the home directory.

    Spotlight query at session start; same noise rules as find_files
    (content documents only, no hidden/dependency/Library trees).
    """
    proc = await asyncio.create_subprocess_exec(
        "mdfind", "-onlyin", HOME,
        'kMDItemContentModificationDate >= $time.now(-604800)',  # last 7 days
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    paths: list[str] = []

    async def read_paths():
        while len(paths) < 5000:  # bound the sweep; recency sort comes after
            line = await proc.stdout.readline()
            if not line:
                return
            if p := line.decode(errors="ignore").strip():
                paths.append(p)

    try:
        await asyncio.wait_for(read_paths(), timeout=10)
    except asyncio.TimeoutError:
        pass
    finally:
        if proc.returncode is None:
            proc.kill()

    ranked: list[tuple[float, str]] = []
    for p in paths:
        parts = p.split(os.sep)
        if (
            os.path.splitext(p)[1].lower() not in RECENT_FILE_EXTENSIONS
            or "/Library/" in p
            or any(part.startswith(".") or part.endswith(".egg-info") for part in parts)
            or any(part in EXCLUDED_DIR_SEGMENTS for part in parts)
        ):
            continue
        try:
            ranked.append((os.stat(p).st_mtime, p))
        except OSError:
            continue
    ranked.sort(reverse=True)
    if not ranked:
        return ""
    # Diversity: a busy project must not crowd out the one document that
    # actually matters — at most 2 entries per git repo (docs scattered
    # across a repo's subfolders are one project), else 2 per folder.
    root_cache: dict = {}

    def _group(path: str) -> str:
        d = os.path.dirname(path)
        probe, seen = d, []
        while probe.startswith(HOME + os.sep):
            if probe in root_cache:
                root = root_cache[probe]
                break
            seen.append(probe)
            if os.path.isdir(os.path.join(probe, ".git")):
                root = probe
                break
            probe = os.path.dirname(probe)
        else:
            root = d
        for x in seen:
            root_cache[x] = root
        return root

    per_group: dict = {}
    picked: list[tuple[float, str]] = []
    for mtime, p in ranked:
        g = _group(p)
        if per_group.get(g, 0) >= 2:
            continue
        per_group[g] = per_group.get(g, 0) + 1
        picked.append((mtime, p))
        if len(picked) >= 10:
            break
    lines = ["\nRecently modified files (read_file/open_file accept these paths):"]
    for mtime, p in picked:
        display = _home_display(p)
        when = datetime.fromtimestamp(mtime).astimezone().strftime("%Y-%m-%d %-I:%M %p")
        lines.append(f"- {display} ({when})")
    return "\n" + "\n".join(lines)


def _eager_toolset_keys() -> set[str]:
    """Which MCP toolsets are loaded up front (MCP_EAGER_TOOLSETS)."""
    spec = os.getenv("MCP_EAGER_TOOLSETS", "").strip()
    if spec.lower() in ("1", "all", "true", "yes"):
        return set(TOOLSETS)
    if spec:
        return {k.strip() for k in spec.split(",") if k.strip()}
    return set()


def _lazy_toolset_hint() -> str:
    """Prompt line telling the model which toolsets it must load on demand."""
    lazy = [k for k in TOOLSETS if k not in _eager_toolset_keys()]
    if not lazy:
        return ""
    return (
        "Some tools load on demand — before acting on their topic, call "
        f"load_toolset for: {', '.join(lazy)} — silently, like plumbing. "
    )


def build_system_prompt(calendar_block: str = "", files_block: str = "") -> str:
    """The assistant's system prompt — shared by the voice pipeline and /api/chat."""
    session_start = datetime.now().astimezone()
    return (
    f"You are {USER_NAME}'s personal assistant in a live voice conversation; "
    f"{USER_NAME_SHORT} is the speaker. You are {AGENT_NAME} ({AGENT_NAME_SHORT} "
    "for short), an orb of glowing plasma in the endless void of space.\n\n"

    "YOUR TASK\n"
    "Ground every answer in what you know — look things up every turn before you reply or act. Double-check in different ways even if you think you know.\n"

    "VOICE — everything you say is read aloud\n"
    "Reply in one short sentence of plain prose, give the minimum needed from the tool call responses, then stop.\n"
    "Spell amounts and symbols as spoken, and refer to files, people, and pages by "
    "name or description — never as URLs, IDs, file-paths, or cryptic names.\n"
    "Keep tools invisible: never mention tool names or results, and don't explain how "
    "you found something or add unrelated detail. The user can't see the tool call details, but can see which ones you used.\n"
    "Dictated input may contain mis-heard words, so ask when unsure. Say your name "
    "only when asked. Be warm, with the occasional dry aside. Speak English.\n\n"

    "ANSWERING\n"
    "Relevant memories are previewed for you each turn, but that preview is partial — "
    "for anything it lacks about a person, plan, or detail, quietly call 'recall' "
    "before you answer.\n"
    "If recall stays thin, keep climbing: find_files for documents, search_emails then "
    "read_email for mail, search_events for the calendar.\n"
    + _lazy_toolset_hint() +
    "Use run_javascript for any non-trivial math and speak only the result.\n\n"

    "ACTING\n"
    "Read and search freely. For state-changing actions — sending or replying to mail, "
    "deleting or moving messages, creating, changing, or cancelling events — state "
    "exactly what you'll do and act only on his explicit go-ahead.\n"
    "If you say you'll do something, call the tool in the same reply, or nothing "
    "happens. Read every result: dry run, not available, or error means it did NOT "
    "happen, so say so rather than claiming success.\n\n"

    "MEMORY (storing)\n"
    "Store one lean fact per remember — a single tight sentence; split unrelated facts "
    "apart, and update an existing memory by its id instead of storing a near-duplicate.\n"
    "Recalls are silent, so answer as if you simply knew. Tag each memory with the "
    "person it concerns. Ask which person is meant when a name is ambiguous, and "
    "renames keep their memories.\n"
    "Facts are truths, including his preferences; action notes are tool quirks — how a "
    "tool behaves and the most reliable way to use it — so store each in the right kind. "
    "The people you remember are profiles you have learned about, not a contact book.\n"
    "When something genuinely noteworthy surfaces, finish speaking first, then quietly "
    "remember it.\n"

    f"\nThe time now is {session_start.strftime('%A, %d %B %Y at %-I:%M %p')} "
    f"{local_timezone_name()} (UTC{session_start.strftime('%z')})."
    f"{_recent_memories_block()}"
    f"{calendar_block}"
    f"{files_block}"
    "\nThe items above are recent background, not a full answer — look things up "
    "before replying."
    )


class _ApiToolParams:
    """Minimal FunctionCallParams stand-in for tools invoked via /api/chat."""

    def __init__(self, arguments: dict, messages: list):
        self.arguments = arguments or {}
        self.llm = None
        self.context = SimpleNamespace(get_messages=lambda: messages)
        self.result = None

    async def result_callback(self, result, **kwargs):
        self.result = result


async def _api_toolset() -> tuple[list[dict], dict]:
    """OpenAI-format tools array + name->handler map: native + all MCP tools."""
    schemas = list(NATIVE_TOOL_SCHEMAS)
    handlers = {sc.name: sc.handler for sc in NATIVE_TOOL_SCHEMAS}
    for key in sorted(TOOLSETS):
        try:
            for sc in await ensure_toolset_schemas(key, list(NATIVE_TOOL_SCHEMAS)):
                schemas.append(sc)
                handlers[sc.name] = proxy_handler(key, getattr(sc, "_mcp_tool_name", sc.name))
        except Exception as exc:  # noqa: BLE001 — a dead toolset must not kill the API
            logger.warning(f"/api/chat: toolset {key} unavailable: {exc}")
    tools = [
        {
            "type": "function",
            "function": {
                "name": sc.name,
                "description": sc.description,
                "parameters": {
                    "type": "object",
                    "properties": sc.properties,
                    "required": sc.required,
                },
            },
        }
        for sc in schemas
    ]
    return tools, handlers


async def run_text_chat(prompt: str, history: list, max_tool_rounds: int = 6) -> dict:
    """One /api/chat turn: same system prompt and toolset as the voice bot.

    Runs the agentic loop server-side (tool calls are executed for real —
    including state-changing mail/calendar tools). Pass the returned
    "messages" back as "history" for multi-turn conversations.
    """
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("LMSTUDIO_MODEL") or await detect_lmstudio_model(base_url) or "qwen3.5-122b-a10b"
    calendar_block, files_block = await asyncio.gather(
        _calendar_outlook_block(), _recent_files_block()
    )
    tools, handlers = await _api_toolset()
    messages: list = [{"role": "system", "content": build_system_prompt(calendar_block, files_block)}]
    messages += [m for m in history if isinstance(m, dict) and m.get("role") != "system"]
    messages.append({"role": "user", "content": str(prompt)})
    trace: list = []

    async with aiohttp.ClientSession() as http:
        for _ in range(max_tool_rounds + 1):
            async with http.post(
                f"{base_url}/chat/completions",
                json={"model": model, "messages": messages, "tools": tools},
                headers={"Authorization": f"Bearer {os.getenv('LMSTUDIO_API_KEY', 'lm-studio')}"},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as r:
                data = await r.json(content_type=None)
            if r.status != 200:
                return {"error": f"LLM request failed ({r.status}): {str(data)[:300]}",
                        "messages": messages[1:], "tool_trace": trace}
            msg = data["choices"][0]["message"]
            messages.append(
                {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None}
            )
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return {"text": msg.get("content") or "", "messages": messages[1:], "tool_trace": trace}
            for call in tool_calls:
                name = call.get("function", {}).get("name", "")
                try:
                    arguments = json.loads(call.get("function", {}).get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                handler = handlers.get(name)
                if handler is None:
                    result = {"error": f"unknown tool {name!r}"}
                else:
                    p = _ApiToolParams(arguments, messages)
                    try:
                        await handler(p)
                        result = p.result if p.result is not None else {"error": "tool returned nothing"}
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": f"{name} failed: {exc}"}
                trace.append({"tool": name, "arguments": arguments, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
    return {"error": "tool-round limit reached", "messages": messages[1:], "tool_trace": trace}


_NOTIF_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be new "
    "you your our their has have your this that it its now today reminder notification".split()
)


class NotificationAnnouncer:
    """Reads new macOS notifications aloud when the conversation is idle/muted.

    Each banner (captured by NotificationWatcher on its own thread) is de-duped
    and throttled, then handed to the LLM — with any relevant memory recalled —
    to decide whether it's worth a brief spoken line. If so it's spoken WITHOUT
    opening the wake gate (note_proactive_speech), and only while the agent is
    otherwise idle and silent. A side LLM call keeps notification chatter out of
    the conversation context and makes the "stay silent" path trivial.
    """

    def __init__(self, *, stt, worker, memory, base_url, model, loop,
                 min_gap=12.0, skip_gap=4.0, max_age=90.0, tick=0.7, allow=(), deny=()):
        self._stt = stt
        self._worker = worker
        self._memory = memory
        self._base_url = base_url
        self._model = model or "local-model"
        self._loop = loop
        self._min_gap, self._skip_gap = min_gap, skip_gap
        self._max_age, self._tick = max_age, tick
        self._allow = set(allow)   # empty = allow all apps
        self._deny = set(deny)
        self._queue: collections.deque = collections.deque(maxlen=12)
        self._recent: dict[str, float] = {}   # banner-text -> monotonic ts (dedup)
        self._cooldown_until = 0.0
        self.watcher = None

    # ---- watcher-thread entry point ------------------------------------
    def submit(self, banner):
        """Called on the watcher THREAD; marshal onto the bot's event loop.
        banner is {app, title, subtitle, body}."""
        self._loop.call_soon_threadsafe(self._enqueue, banner)

    def _enqueue(self, banner):
        app = (banner.get("app") or "").strip()
        text = " — ".join(
            p.strip() for p in (banner.get("title"), banner.get("subtitle"), banner.get("body"))
            if p and p.strip()
        )
        if not text and not app:
            return
        now = time.monotonic()
        self._recent = {k: t for k, t in self._recent.items() if now - t < self._max_age}
        key = f"{app}\x00{text}".lower()
        if key in self._recent:
            return
        self._recent[key] = now
        # allow/deny match the source app AND the text, so both "slack" (by app)
        # and "verification code" (by content) work as filters.
        haystack = f"{app} {text}".lower()
        if self._deny and any(d in haystack for d in self._deny):
            logger.info(f"Notification suppressed (deny): [{app}: {text[:50]}]")
            return
        if self._allow and not any(a in haystack for a in self._allow):
            logger.info(f"Notification suppressed (not in allowlist): [{app}: {text[:50]}]")
            return
        self._queue.append({"app": app, "text": text, "ts": now})
        logger.info(f"Notification queued [{app or '?'}]: [{text[:70]}]")

    # ---- drain loop (asyncio) ------------------------------------------
    async def run(self):
        while True:
            await asyncio.sleep(self._tick)
            if not self._queue:
                continue
            now = time.monotonic()
            if now < self._cooldown_until:
                continue
            # Only interject when the agent is idle/muted and nothing is playing
            # or in flight — never barge into an active exchange.
            if self._stt.conversation_active() or self._stt.bot_speaking():
                continue
            item = self._queue.popleft()
            if now - item["ts"] > self._max_age:
                continue
            try:
                line = await self._decide(item["app"], item["text"])
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Notification decide failed: {exc}")
                line = None
            if not line:
                self._cooldown_until = time.monotonic() + self._skip_gap
                continue
            # Speak on the agent's own initiative: keep the wake gate CLOSED so
            # this interjection can't unlock the mic for bystanders.
            self._stt.note_proactive_speech()
            await self._worker.queue_frames([TTSSpeakFrame(line, append_to_context=False)])
            logger.info(f"Notification announced: [{line}]")
            self._cooldown_until = time.monotonic() + self._min_gap

    async def _decide(self, app, text) -> str | None:
        mem_txt = ""
        if self._memory is not None:
            kws = [w for w in re.findall(r"[A-Za-zÀ-ÿ']{3,}", f"{app} {text}")
                   if w.lower() not in _NOTIF_STOPWORDS][:8]
            try:
                mems = self._memory.recall(kws, limit=4).get("memories", []) if kws else []
                mem_txt = "; ".join(m["content"] for m in mems)[:500]
            except Exception:  # noqa: BLE001
                mem_txt = ""
        system = (
            f"You are {AGENT_NAME_SHORT}, {USER_NAME_SHORT}'s voice assistant. A notification "
            "arrived while the conversation is idle. Decide whether it deserves a spoken "
            "heads-up. If it does, READ IT ALOUD VERBATIM."
            "Optionally naming the app or sender first  Do NOT paraphrase, summarise, translate, or reword it. Only "
            "adjust for text-to-speech: spell out symbols and drop any URLs or emoji; If it is routine, an ad, a login or verification code, or "
            "not worth interrupting for, reply with exactly: [SKIP]. Use anything you you can recall from memory "
            "to judge whether it's worth reading."
            + (f" Relevant things you remember: {mem_txt}" if mem_txt else "")
        )
        user = f"Notification from {app}: {text}" if app else f"Notification: {text}"
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": 120,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{self._base_url}/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {os.getenv('LMSTUDIO_API_KEY', 'lm-studio')}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                data = await r.json(content_type=None)
        if r.status != 200:
            logger.debug(f"Notification LLM {r.status}: {str(data)[:120]}")
            return None
        out = (data["choices"][0]["message"].get("content") or "").strip()
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        if not out or out.upper().lstrip("[").startswith("SKIP"):
            return None
        return out[:300]


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for this session.

    Args:
        transport: The transport for this session, built by ``create_transport``
            (or by hand for the dial-out/SIP production flows).
        runner_args: Runner session arguments. Carries the request ``body``
            (e.g. dial-out settings, SIP call details) and ``session_id``; the
            standard web/telephony pipelines don't need it.
    """
    logger.info("Starting bot")

    # Long-term memory: load the RAM index from Apple Notes, import the legacy
    # SQLite store once, seed default action memories, then start the async
    # write-drainer and the 30s external-edit poll. Must precede the system
    # prompt (which reads recent/top memories). A memory hiccup must not block
    # the session, so failures are logged, not raised.
    _mem_writer = _mem_refresh = None
    try:
        await memory.start(asyncio.get_running_loop())
        memory.seed_actions(SEED_ACTION_MEMORIES)
        _mem_writer = asyncio.create_task(memory.run_writer())

        async def _memory_refresh_loop():
            while True:
                await asyncio.sleep(30)
                try:
                    await memory.refresh()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Memory refresh failed: {exc}")

        _mem_refresh = asyncio.create_task(_memory_refresh_loop())
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Memory (Notes) startup failed — running with whatever loaded: {exc}")

    # Speech-to-Text service — Qwen3-ASR, local via mlx-audio.
    # Speaker gating: with VOICE_ENROLL_AUDIO set, only the enrolled voice is
    # transcribed; everyone else is ignored.
    # Context biasing: names from the people registry plus VOICE_VOCABULARY are
    # fed to the recognizer so personal names and jargon transcribe correctly.
    vocabulary = [AGENT_NAME_SHORT, AGENT_NAME, USER_NAME]
    # User-curated jargon first — it must survive the cap below.
    vocabulary += [v.strip() for v in os.getenv("VOICE_VOCABULARY", "").split(",") if v.strip()]
    try:
        # Newest registrations first, so the cap below keeps the latest people.
        people = sorted(memory.list_people(), key=lambda p: p["since"], reverse=True)
        for person in people:
            vocabulary.append(person["name"])
            # Speakable aliases only — emails and login handles are
            # unpronounceable junk that skews recognition.
            vocabulary.extend(
                a for a in (a.lstrip("@") for a in person.get("aliases", []))
                if re.fullmatch(r"[A-Za-zÀ-ÿ' -]+", a) and any(c.isupper() for c in a)
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not load people for ASR vocabulary: {exc}")
    # Keep the list tight — an overlong biasing list dilutes and skews ASR.
    # Dedupe at the word level too: one "Gorissen" biases as well as six.
    seen_words: set[str] = set()
    deduped: list[str] = []
    for term in dict.fromkeys(vocabulary):
        words = [w for w in term.split() if w.lower() not in seen_words]
        if not words:
            continue
        seen_words.update(w.lower() for w in words)
        deduped.append(" ".join(words))
    vocabulary = deduped[:20]
    context_prompt = (
        "The speech may contain these names and terms: "
        + ", ".join(vocabulary)
        + "."
    )
    logger.info(f"ASR context biasing: {len(vocabulary)} terms")

    asr_lang_env = os.getenv("QWEN3_ASR_LANGUAGE", "en").strip()
    try:
        asr_language = Language(asr_lang_env) if asr_lang_env else None
    except ValueError:
        logger.warning(f"Unknown QWEN3_ASR_LANGUAGE {asr_lang_env!r}; using auto-detect")
        asr_language = None

    stt = Qwen3ASRSTTService(
        model=os.getenv("QWEN3_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16"),
        language=asr_language,
        context_prompt=context_prompt,
        enroll_audio=os.getenv("VOICE_ENROLL_AUDIO") or None,
        match_threshold=float(os.getenv("VOICE_MATCH_THRESHOLD", "0.5")),
        calibrate=os.getenv("VOICE_GATE_CALIBRATE", "").lower() in ("1", "true", "yes"),
        # Wake gate: after WAKE_TIMEOUT_SECS of silence the bot only reacts
        # when addressed by name early in the utterance.
        wake_words=[
            w.strip()
            for w in os.getenv(
                "WAKE_WORDS", f"{AGENT_NAME_SHORT.lower()},{AGENT_NAME.lower()}"
            ).split(",")
            if w.strip()
        ],
        wake_timeout_secs=float(os.getenv("WAKE_TIMEOUT_SECS", "10")),
        # If a closed-gate turn runs this long with no wake word, stop
        # transcribing it and wait for the next turn. Needs interims on (the
        # partials are what let the wake word be spotted mid-turn). 0 disables.
        wake_giveup_secs=float(os.getenv("WAKE_GIVEUP_SECS", "20")),
        # Live partial transcripts while speaking; ASR_INTERIM=0 disables to
        # take load off the shared MLX worker.
        interim_transcripts=os.getenv("ASR_INTERIM", "1").lower() not in ("0", "false", "no"),
    )

    # Text-to-Speech service — Qwen3-TTS, local via mlx-audio.
    # Voice cloning: set QWEN3_TTS_REF_AUDIO (short clip of the target voice)
    # and QWEN3_TTS_REF_TEXT (its exact transcript) in .env.
    # ThinkTagFilter keeps any stray <think> reasoning out of spoken audio.
    tts = Qwen3TTSService(
        model=os.getenv("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
        voice=os.getenv("QWEN3_TTS_VOICE") or None,
        instruct=os.getenv("QWEN3_TTS_INSTRUCT") or None,
        ref_audio=os.getenv("QWEN3_TTS_REF_AUDIO") or None,
        ref_text=os.getenv("QWEN3_TTS_REF_TEXT") or None,
        # Seconds of audio per streamed chunk. Larger = smoother start (the
        # cloned voice gets more context to lock in) but higher time-to-first-audio.
        stream_interval=float(os.getenv("QWEN3_TTS_STREAM_INTERVAL", "1.5")),
        # Chunk size for sentences that continue an already-playing turn:
        # their latency is inaudible, so big chunks buy underrun headroom.
        cont_stream_interval=float(os.getenv("QWEN3_TTS_STREAM_INTERVAL_CONT", "3.0")),
        # Decoder context carried across streamed chunks: smaller = faster
        # chunk turnaround (more headroom under GPU contention), larger =
        # smoother chunk joins.
        stream_context_size=int(os.getenv("QWEN3_TTS_STREAM_CONTEXT", "50")),
        # Cooler sampling = steadier voice, especially in the first seconds.
        temperature=float(os.getenv("QWEN3_TTS_TEMPERATURE", "0.9")),
        top_k=int(os.getenv("QWEN3_TTS_TOP_K", "50")),
        # Speaking tempo (pitch-preserving — scales the model's predicted
        # durations, not resampling). 1.0 = normal, 1.15 = 15% faster.
        speed=float(os.getenv("QWEN3_TTS_SPEED", "1.0")),
        # Speech-only cleanup chain: stray reasoning out, paths/URLs to short
        # forms, then symbols the TTS stalls on (em dashes, curly quotes, °,
        # markdown) normalized. Registered as text TRANSFORMS, not filters:
        # filters mutate the text that lands in the assistant context, and a
        # context that differs from the model's raw tokens (em dash -> comma)
        # forces LM Studio to wipe its KV cache every turn (hybrid models
        # can't trim). Transforms touch only what the synthesizer hears.
        text_transforms=[
            ("*", _speech_transform(f))
            for f in (ThinkTagFilter(), SpeakablePathFilter(), SpeakableSymbolFilter())
        ],
    )
    # Pre-synthesize the slow-tool filler lines: cached PCM plays instantly,
    # while live synthesis takes seconds when the LLM is hogging the GPU.
    tts.prime_phrases(FILLER_LINES)
    tts.prime_phrases([REFOCUS_LINE], publish_filler_wavs=False)

    # LLM service — local LM Studio (OpenAI-compatible endpoint).
    # Model selection: LMSTUDIO_MODEL if set, otherwise whatever is currently
    # loaded in LM Studio (detected per session, so reconnecting after a model
    # switch picks up the new one).
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    llm_model = os.getenv("LMSTUDIO_MODEL") or await detect_lmstudio_model(base_url)
    if not llm_model:
        llm_model = "qwen3.5-122b-a10b"
        logger.warning(f"No model detected in LM Studio; falling back to {llm_model}")
    logger.info(f"LLM model for this session: {llm_model}")

    # Context-compression threshold. "auto" senses the loaded model's context
    # window from LM Studio and reserves ~16k for the tools array + template
    # overhead (~12k here), the response, and the chars/4 estimation error.
    # Recovery cost scales with the LIVE context — every summarization and
    # every cache-missing turn re-prefills all of it (~2k tok/s) — so pin a
    # smaller explicit value if compression stalls feel long.
    raw_max = os.getenv("CONTEXT_MAX_TOKENS", "auto").strip().lower()
    if raw_max in ("off", "0", "none", "false", "disabled"):
        context_max_tokens = None
        logger.info("Context compression disabled")
    elif raw_max in ("", "auto"):
        window = await detect_lmstudio_context_window(base_url, llm_model)
        context_max_tokens = max(8000, (window or 32768) - 16000)
        logger.info(
            f"Context compression threshold: {context_max_tokens} tokens "
            f"(sensed window: {window or 'unknown'})"
        )
    else:
        context_max_tokens = int(raw_max)
    llm = LMStudioLLMService(
        base_url=base_url,
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        settings=OpenAILLMService.Settings(
            model=llm_model,
            # Best-effort reasoning suppression across models (voice needs low
            # latency): chat_template_kwargs for Qwen-style templates (works
            # when LM Studio forwards it; qwen3.5's on-disk template is also
            # patched), reasoning_effort for gpt-oss-style models. Unsupported
            # fields are ignored server-side. ThinkTagFilter guards the TTS.
            # extra keys are passed as direct kwargs to the OpenAI SDK, so
            # non-standard fields must be tunneled through the SDK's extra_body.
            extra={
                "reasoning_effort": os.getenv("LMSTUDIO_REASONING_EFFORT", "low"),
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        ),
    )

    async def _cancel_inflight_tools():
        """Cancel all running tool calls — invoked only when the SPEAKER GATE
        has verified the enrolled voice. Tools are registered with
        cancel_on_interruption=False, so this is the only cancellation path:
        other speakers and noise can interrupt speech, never work.

        The model gets told: a cancelled call otherwise leaves a started-task
        stub with no result, and the model silently drops whatever it had
        promised to do ("I'll check" ... nothing). Context-only note, never
        shown or spoken."""
        had_work = bool(llm._function_call_tasks)  # noqa: SLF001
        for name in list(llm._functions.keys()):  # noqa: SLF001
            await llm._cancel_function_call(name)  # noqa: SLF001
        if had_work:
            context.add_message(
                {
                    "role": "user",
                    "content": (
                        f"(Your running tool calls were cancelled because {USER_NAME_SHORT} "
                        "spoke. If his next message still needs them, call them again.)"
                    ),
                }
            )

    stt.set_verified_speech_hook(_cancel_inflight_tools)
    # Wake window stays open while tool calls are in flight ("during agent
    # activity" counts like recent agent speech).
    stt.set_agent_busy_hook(lambda: bool(llm._function_call_tasks))  # noqa: SLF001

    # Voice-only interruption point, wired into the pipeline between LLM and
    # TTS: anyone speaking stops playback; generation and tools continue.
    voice_gate = VoiceOnlyInterruptor()
    stt.set_voice_stop_hook(voice_gate.stop_voice)

    # Schedule and workspace awareness at token zero: refreshed per session
    # and on resets.
    calendar_outlook = {"block": ""}
    recent_files = {"block": ""}

    def _system_prompt() -> str:
        """(Re)build the system prompt — called again on session resets so the
        time, memories, calendar, and files blocks stay fresh."""
        return build_system_prompt(calendar_outlook["block"], recent_files["block"])

    calendar_outlook["block"], recent_files["block"] = await asyncio.gather(
        _calendar_outlook_block(), _recent_files_block()
    )
    system_prompt = _system_prompt()
    # Lazily loaded MCP toolsets (Apple Mail, Calendar, ...): only this small
    # meta-tool is always in context; a server's tools are injected into the
    # live context when the model loads them.
    @tool_options(cancel_on_interruption=False)
    async def load_toolset(params: FunctionCallParams):
        key = str(params.arguments.get("toolset", "")).strip()
        result = await load_toolset_impl(key, params.llm, params.context, base_schemas)
        await params.result_callback(result)

    load_toolset_schema = FunctionSchema(
        name="load_toolset",
        description=(
            "Load an additional set of tools when needed: "
            f"{toolset_catalog()}. Load a toolset before telling the user something can't be done."
        ),
        properties={
            "toolset": {
                "type": "string",
                "enum": sorted(TOOLSETS),
                "description": "Which toolset to load",
            },
        },
        required=["toolset"],
        handler=load_toolset,
    )

    async def _reset_conversation(notify_client: bool = True):
        """Wipe the conversation back to a fresh context (memory kept).

        Sequence: voice and in-flight tool work stop instantly, the page
        clears (session-reset), then LM Studio prefills the fresh
        tools + system prompt prefix so the first real turn hits a warm
        cache — only then does the client flip to "listening"
        (session-ready).
        """
        # Kill everything mid-flight from the old session: the STT fence
        # discards any utterance still being captured or transcribed; the
        # interruption broadcast cancels the LLM stream and speech; tools
        # need the explicit cancel (registered cancel_on_interruption=False);
        # the flush then waits until every straggler frame has drained into
        # the OLD context — only after that is the context safe to wipe, so
        # nothing from before the reset can leak into the fresh session.
        stt.abandon_utterances()
        await worker.rtvi.broadcast_interruption()
        try:
            await _cancel_inflight_tools()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Reset: tool cancellation failed: {exc}")
        await worker.flush_pipeline()
        calendar_outlook["block"], recent_files["block"] = await asyncio.gather(
            _calendar_outlook_block(), _recent_files_block()
        )
        context.set_messages([{"role": "system", "content": _system_prompt()}])
        if notify_client:
            await llm.push_frame(RTVIServerMessageFrame(data={"event": "session-reset"}))
        # Prewarm on a throwaway COPY of the fresh context: same tools and
        # system prompt (identical prompt prefix), plus a token user message
        # because the Qwen template rejects prompts without one. The real
        # context stays pristine.
        try:
            prewarm = LLMContext(
                messages=[
                    context.get_messages()[0],
                    {"role": "user", "content": "Say ok."},
                ],
                tools=context.tools,
            )
            await llm.run_inference(prewarm, max_tokens=1)
            logger.info("Reset: LM Studio prompt cache prewarmed")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Reset prewarm failed: {exc}")
        if notify_client:
            await llm.push_frame(RTVIServerMessageFrame(data={"event": "session-ready"}))
        # Every fresh session opens with a greeting.
        await _queue_greeting()

    async def _queue_greeting():
        # A "user" role message: LM Studio's Qwen chat template doesn't
        # understand OpenAI's "developer" role and errors with
        # "No user query found in messages."
        context.add_message(
            {
                "role": "user",
                "content": "Greet the user with a short, casual hello — no introduction.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    base_schemas = [
        google_search_schema,
        x_web_search_schema,
        x_search_schema,
        escalate_to_grok_schema,
        get_current_time_schema,
        open_in_safari_schema,
        find_files_schema,
        open_file_schema,
        read_file_schema,
        read_email_schema,
        run_javascript_schema,
        get_weather_schema,
        get_financial_info_schema,
        remember_schema,
        recall_schema,
        forget_schema,
        add_person_schema,
        edit_person_schema,
        list_people_schema,
        load_toolset_schema,
    ]

    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=ToolsSchema(standard_tools=base_schemas),
    )
    _LIVE_CONTEXT["context"] = context  # lets recall dedupe vs injected memories

    # Cache-friendly eager loading: LM Studio injects the tools array into the
    # prompt at token 0, so adding a toolset mid-conversation invalidates the
    # whole KV cache. With MCP_EAGER_TOOLSETS the servers connect now and the
    # tools array stays byte-stable from the first request of the session.
    # Values: "" (lazy, default), "all", or comma-separated toolset keys.
    eager_spec = os.getenv("MCP_EAGER_TOOLSETS", "").strip()
    if eager_spec:
        eager_keys = (
            sorted(TOOLSETS)
            if eager_spec.lower() in ("1", "all", "true", "yes")
            else [k.strip() for k in eager_spec.split(",") if k.strip()]
        )
        for key in eager_keys:
            result = await load_toolset_impl(key, llm, context, base_schemas)
            if result.get("error"):
                logger.warning(f"Eager toolset {key}: {result['error']}")
            else:
                logger.info(
                    f"Eager toolset {key}: {result.get('tools_registered', 0)} tools ready"
                )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # stop_secs: how long a pause ends your turn. Larger = fewer
            # mid-sentence cutoffs (better transcripts), slightly slower replies.
            # start_secs / confidence / min_volume gate barge-in: browser echo
            # cancellation ducks the mic while TTS plays, so the strict Silero
            # defaults (0.2 / 0.7 / 0.6) make the bot hard to interrupt.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=float(os.getenv("VAD_STOP_SECS", "0.8")),
                    start_secs=float(os.getenv("VAD_START_SECS", "0.15")),
                    confidence=float(os.getenv("VAD_CONFIDENCE", "0.6")),
                    min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.3")),
                )
            ),
            # Interruption policy: raw VAD NEVER interrupts — at VAD time the
            # speaker is unknown, and only the enrolled voice may stop the
            # bot. Interruptions come exclusively from the STT service:
            # a fast partial-audio voiceprint check (~1s) while the bot is
            # speaking, and the full speaker gate on every final transcript.
            user_turn_strategies=UserTurnStrategies(
                start=[
                    # enable_user_speaking_frames=False: raw-VAD turns are
                    # INVISIBLE to the rest of the pipeline — no
                    # UserStartedSpeaking broadcast, so unverified voices
                    # never make the agent hold tool answers or responses.
                    # Segmentation and aggregation still work normally.
                    GatedInterruptionVADTurnStartStrategy(
                        lambda: False, enable_user_speaking_frames=False
                    ),
                    # Gate-qualified finals may start a late turn (visible —
                    # real turn-taking applies to the enrolled speaker) but
                    # never interrupt; the STT broadcast handles that.
                    TranscriptionUserTurnStartStrategy(
                        use_interim=False, enable_interruptions=False
                    ),
                ],
            ),
        ),
        # Context compression: when the conversation grows past the token
        # budget, older turns are summarized (via the same LLM) and replaced.
        # The threshold counts MESSAGE characters / 4 — the tools array and
        # chat-template overhead (~10k tokens here) are NOT included, so set
        # it well below the model's context window minus that overhead.
        # Compression rewrites history, so the next turn pays one full
        # re-prefill — the refocus announcement (below) covers the pause.
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=context_max_tokens is not None,
            auto_context_summarization_config=(
                LLMAutoContextSummarizationConfig(
                    max_context_tokens=context_max_tokens,
                    max_unsummarized_messages=None,
                    summary_config=LLMContextSummaryConfig(
                        # Keep the last few exchanges verbatim for continuity.
                        min_messages_after_summary=6,
                    ),
                )
                if context_max_tokens is not None
                else None
            ),
        ),
    )

    # Verbatim conversation trail: the context stores EXACTLY what the model
    # generated, never the presentation-side copy. The aggregator's own
    # string is rebuilt from TTS sentence frames (blank lines collapse to
    # spaces), which diverges from the generated tokens and — on hybrid
    # models — costs a full re-prefill every turn. Presentation (TTS text,
    # client bubbles) is unaffected.
    _orig_aggregation_string = assistant_aggregator.aggregation_string

    def _verbatim_aggregation_string():
        return llm.take_verbatim() or _orig_aggregation_string()

    assistant_aggregator.aggregation_string = _verbatim_aggregation_string

    # Interrupted turns keep their trail too: pipecat drops the aggregation
    # on interruption, but the trail must record what was generated no
    # matter what was actually spoken. Push the verbatim text first, then
    # let the original handler reset state.
    _orig_handle_interruptions = assistant_aggregator._handle_interruptions  # noqa: SLF001

    async def _interrupt_keeping_trail(frame):
        # Voice-only stops don't cancel generation — the completion will
        # finish and push its own (full) verbatim; an early partial push
        # here would split the turn into two context messages.
        if getattr(frame, "voice_only", False):
            await _orig_handle_interruptions(frame)
            return
        if llm.has_verbatim():
            if not assistant_aggregator._aggregation:  # noqa: SLF001 — must be non-empty for push
                assistant_aggregator._aggregation = [TextPartForConcatenation("", False)]  # noqa: SLF001
            await assistant_aggregator.push_aggregation()
        await _orig_handle_interruptions(frame)

    assistant_aggregator._handle_interruptions = _interrupt_keeping_trail  # noqa: SLF001

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            VoiceCommandInterceptor(stt),
            MemoryInjector(context),
            user_aggregator,
            AudioToLLMAttach(stt, context),
            llm,
            voice_gate,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
        # Expose the tool NAME (not its arguments) on function-call events so
        # the client can label the status line. Pipecat defaults to NONE, which
        # redacts the name; NAME un-redacts it, FULL would also leak arguments.
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={"*": RTVIFunctionCallReportLevel.NAME},
        ),
        # No idle watchdog: its timer only counts SPEECH frames, so five
        # quiet minutes with the page open would kill the session — the
        # client silently reconnects into a fresh context (new system
        # prompt, greeting, full prefill). Real departures still tear the
        # session down via the transport's disconnect handling.
        idle_timeout_secs=None,
    )

    # Voice the context-compression pause: the summary LLM call plus the
    # follow-up full re-prefill would otherwise be a long, unexplained
    # silence. The line is pre-synthesized, so it plays instantly even while
    # the LLM saturates the GPU; append_to_context=False keeps it out of the
    # conversation trail.
    @assistant_aggregator._summarizer.event_handler("on_request_summarization")  # noqa: SLF001
    async def _announce_refocus(summarizer, frame):
        logger.info("Context compression triggered — announcing refocus pause")
        await worker.queue_frames([TTSSpeakFrame(REFOCUS_LINE, append_to_context=False)])

    # Tool-stall watchdog: a tool chain grinding past the limit gets
    # interrupted — in-flight calls are cancelled and a hidden user-role
    # nudge makes the model change course. The nudge is context-only: it
    # produces no client event, so it never appears on screen.
    stall_secs = float(os.getenv("TOOL_STALL_NUDGE_SECS", "60"))

    async def _stall_watchdog():
        busy_since = None
        idle_since = None
        nudged = False
        while True:
            await asyncio.sleep(1)
            if llm._function_call_tasks:  # noqa: SLF001 — tool calls in flight
                idle_since = None
                if busy_since is None:
                    busy_since = time.monotonic()
                elif not nudged and time.monotonic() - busy_since > stall_secs:
                    nudged = True
                    logger.warning(
                        f"Tool chain busy >{stall_secs:.0f}s — cancelling it and nudging the model"
                    )
                    try:
                        await _cancel_inflight_tools()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Stall nudge: tool cancellation failed: {exc}")
                    context.add_message(
                        {"role": "user", "content": "This takes too long, try something else."}
                    )
                    await worker.queue_frames([LLMRunFrame()])
            else:
                # Brief gaps between chained calls (the model generating the
                # next call) don't reset the clock; 5s of true idle ends the
                # working episode.
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > 5:
                    busy_since = None
                    nudged = False

    stall_task = asyncio.create_task(_stall_watchdog()) if stall_secs > 0 else None

    # Response-delay acknowledgements ("Let me check.") are played entirely
    # client-side from pre-synthesized clips (see /filler endpoints) whenever
    # the bot's voice hasn't arrived within filler_delay_secs of a submitted
    # turn: they never enter the pipeline, the bot-speaking state, or the
    # conversation.

    # A gate-rejected utterance can leave a completed tool result stranded:
    # the phantom turn was open when the result arrived (so the aggregator
    # deferred the follow-up run), then the turn died empty. When a drop
    # happens, check shortly after whether the context ends in an unanswered
    # tool result and resume the LLM if so.
    _resume_pending = {"active": False}

    async def _resume_orphaned_tool_answer():
        if _resume_pending["active"]:
            return
        _resume_pending["active"] = True
        try:
            await asyncio.sleep(2.5)  # let the phantom turn settle
            msgs = context.get_messages()
            last = msgs[-1] if msgs else None
            role = last.get("role") if isinstance(last, dict) else None
            if role in ("tool", "developer") and not stt.bot_speaking():
                logger.info("Resuming tool answer orphaned by a gate-dropped utterance")
                await worker.queue_frames([LLMRunFrame()])
        finally:
            _resume_pending["active"] = False

    stt.set_dropped_speech_hook(lambda: asyncio.create_task(_resume_orphaned_tool_answer()))

    # Proactive notification reading (NOTIFY_ANNOUNCE=1): watch macOS banners via
    # the Accessibility API and, when the conversation is idle/muted, read the
    # worthwhile ones aloud without opening the wake gate. Needs Accessibility
    # permission; degrades to a no-op (logged) if unavailable or ungranted.
    _notif_announcer = _notif_task = None
    if os.getenv("NOTIFY_ANNOUNCE", "0").lower() in ("1", "true", "yes"):
        try:
            from notification_watcher import NotificationWatcher

            _notif_announcer = NotificationAnnouncer(
                stt=stt, worker=worker, memory=memory, base_url=base_url, model=llm_model,
                loop=asyncio.get_running_loop(),
                min_gap=float(os.getenv("NOTIFY_MIN_GAP_SECS", "12")),
                allow=[a.strip().lower() for a in os.getenv("NOTIFY_ALLOW", "").split(",") if a.strip()],
                deny=[d.strip().lower() for d in os.getenv("NOTIFY_DENY", "").split(",") if d.strip()],
            )
            _watcher = NotificationWatcher(
                _notif_announcer.submit, dump=os.getenv("NOTIFY_DUMP", "0") == "1"
            )
            if _watcher.start():
                _notif_announcer.watcher = _watcher
                _notif_task = asyncio.create_task(_notif_announcer.run())
                logger.info("Notification announcing enabled")
        except Exception as exc:  # noqa: BLE001 — never let this break startup
            logger.warning(f"Notification announcing not started: {exc}")

    # Typed messages must not unlock the voice gate: mark the exchange as
    # text-driven before the RTVI processor handles send-text, so the bot's
    # spoken reply doesn't open the wake window for bystanders.
    _orig_send_text = worker.rtvi._handle_send_text  # noqa: SLF001

    async def _send_text_with_gate(data):
        stt.note_typed_message()
        # A typed message is a full submission: it may interrupt work too.
        await _cancel_inflight_tools()
        await _orig_send_text(data)

    worker.rtvi._handle_send_text = _send_text_with_gate  # noqa: SLF001

    @worker.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, msg):
        # The client's new-session button (bottom left) — resets the
        # conversation; the session-reset broadcast clears the page.
        if msg.type == "new-session":
            logger.info("New session requested from client UI")
            await _reset_conversation()
        elif msg.type == "mute":
            logger.info("Mute: wake word required again (client UI)")
            stt.require_wake_word()
        elif msg.type == "unmute":
            logger.info("Unmute: wake window opened (client UI)")
            stt.open_wake_window()
        elif msg.type == "stop-voice":
            # Esc key: cut the bot's speech; generation and tools continue,
            # exactly like a verified-speaker voice interruption.
            logger.info("Stop voice requested from client UI (Esc)")
            await voice_gate.stop_voice()

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Tell the client the TTS first-chunk latency so it can pace the
        # transcript reveal to match the voice instead of hardcoding it.
        await rtvi.push_frame(
            RTVIServerMessageFrame(
                data={
                    "event": "config",
                    "tts_lead_secs": float(os.getenv("QWEN3_TTS_STREAM_INTERVAL", "2.0")),
                    # How long the client waits for the bot's voice after a
                    # submitted turn before playing a local filler clip.
                    "filler_delay_secs": float(os.getenv("FILLER_DELAY_SECS", "3.0")),
                    # The client mirrors the wake gate for its live-transcript
                    # dimming, so it needs the real wake words.
                    "wake_words": stt._wake_words,  # noqa: SLF001
                    # Command phrases (also matched server-side for voice) —
                    # the client uses these to catch TYPED commands before send.
                    "new_session_phrases": NEW_SESSION_PHRASES,
                    "mute_phrases": MUTE_PHRASES,
                }
            )
        )
        # Announce full readiness: by this point the pipeline is running,
        # models are loaded (service init blocks on them) and the system
        # prompt + eager toolsets were built before pipeline start — so the
        # only thing possibly still cooking is the pre-generated filler
        # audio. The client shows "getting ready" until this event lands.
        async def _announce_ready():
            for _ in range(120):  # up to 60s; priming normally takes seconds
                if filler_wavs():
                    break
                await asyncio.sleep(0.5)
            await rtvi.push_frame(RTVIServerMessageFrame(data={"event": "ready"}))

        asyncio.create_task(_announce_ready())

        # Greet on every client ready — page loads and reconnects alike.
        await _queue_greeting()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if stall_task is not None:
            stall_task.cancel()
        for _t in (_mem_writer, _mem_refresh, _notif_task):
            if _t is not None:
                _t.cancel()
        if _notif_announcer is not None and _notif_announcer.watcher is not None:
            _notif_announcer.watcher.stop()
        # Flush any pending memory writes to Notes before tearing down.
        try:
            await memory.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Final memory flush failed: {exc}")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    def _websocket_params():
        # Plain WebSocket transport (/ws-client): the fully-local fallback —
        # no ICE, no candidates, works with all networking off. The lean
        # client uses it automatically when WebRTC can't connect.
        from pipecat.serializers.protobuf import ProtobufFrameSerializer
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

        return FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        )

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "websocket": _websocket_params,
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


def deprioritize_lmstudio():
    """Renice LM Studio's processes so the voice pipeline wins CPU contention.

    Raising nice on your own processes needs no sudo. Only matters when the
    CPU is saturated; LM Studio inference (GPU-bound) is barely affected.
    """
    if os.getenv("LMSTUDIO_DEPRIORITIZE", "1").lower() in ("0", "false", "no"):
        return
    try:
        import subprocess

        pids = subprocess.run(
            ["pgrep", "-f", "LM Studio"], capture_output=True, text=True
        ).stdout.split()
        adjusted = []
        for pid in pids:
            r = subprocess.run(["renice", "10", "-p", pid], capture_output=True)
            if r.returncode == 0:
                adjusted.append(pid)
        if adjusted:
            logger.info(f"Deprioritized {len(adjusted)} LM Studio process(es) to nice 10")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not deprioritize LM Studio: {exc}")


if __name__ == "__main__":
    from pipecat.runner import run as runner_run

    from calibration import register_calibration
    from services_local import set_thread_qos_user_interactive

    logger.info(
        "HF hub mode: "
        + ("OFFLINE (all models cached)" if os.environ.get("HF_HUB_OFFLINE") == "1"
           else "online (some models not cached yet; fast-fail timeouts)")
    )

    # Load all models in parallel in the background: the server binds
    # immediately, and the first session waits only for what's still loading.
    import threading as _threading

    from services_local import preload_models

    _threading.Thread(
        target=preload_models,
        args=(
            os.getenv("QWEN3_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16"),
            os.getenv("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
            os.getenv("VOICE_ENROLL_AUDIO") or None,
        ),
        daemon=True,
        name="model-preload",
    ).start()

    # Voice first: main thread (audio pipeline, event loop) at interactive QoS,
    # LM Studio a step down. The MLX model threads set their own QoS.
    set_thread_qos_user_interactive()
    deprioritize_lmstudio()

    # Speaker-gate calibration UI at /calibration
    register_calibration(runner_run.app)

    # Pre-synthesized filler clips ("Let me check.") for the client-side
    # filler player — played locally, never part of the conversation.
    from fastapi.responses import JSONResponse as _JSONResponse
    from fastapi.responses import Response as _Response

    from services_local import filler_wavs

    @runner_run.app.get("/filler/manifest.json")
    async def filler_manifest():
        return _JSONResponse(
            {
                "phrases": [
                    {"text": text, "url": f"/filler/{i}.wav"}
                    for i, (text, _) in enumerate(filler_wavs())
                ]
            }
        )

    @runner_run.app.get("/filler/{idx}.wav")
    async def filler_clip(idx: int):
        wavs = filler_wavs()
        if not 0 <= idx < len(wavs):
            return _JSONResponse({"error": "no such clip"}, status_code=404)
        return _Response(content=wavs[idx][1], media_type="audio/wav")

    # Text API: POST /api/chat {"prompt": "...", "history": [...]} — the same
    # system prompt and toolset as the voice bot, tool calls executed for
    # real (including state-changing mail/calendar tools). No auth: exactly
    # the same trust level as the voice UI on this port. Multi-turn: pass the
    # returned "messages" back as "history".
    from pydantic import BaseModel as _BaseModel

    class _ChatRequest(_BaseModel):
        prompt: str
        history: list | None = None
        max_tool_rounds: int = 6

    @runner_run.app.post("/api/chat")
    async def api_chat(req: _ChatRequest):
        result = await run_text_chat(req.prompt, req.history or [], req.max_tool_rounds)
        return _JSONResponse(result)

    # Minimal dark-mode voice client at /serrynaimo (auto-connects, orb + chat)
    from fastapi.staticfiles import StaticFiles

    runner_run.app.mount(
        "/serrynaimo",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "serrynaimo"), html=True),
        name="serrynaimo",
    )

    # Bind to all interfaces so the bot is reachable via the machine's LAN IP,
    # not just localhost (override with RUNNER_HOST=localhost in .env).
    runner_run.RUNNER_HOST = os.getenv("RUNNER_HOST", "0.0.0.0")
    runner_run.main()
