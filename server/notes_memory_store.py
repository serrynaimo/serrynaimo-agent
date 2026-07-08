"""Apple Notes-backed memory store (drop-in for MemoryStore).

Notes is the source of truth for memory CONTENT; a small SQLite sidecar holds
per-memory STATS (recall counts, timestamps — and future perf logging); a RAM
index is the fast working copy that all reads hit. Layout under a folder named
after the agent:

    {AGENT}/                      <- "General" note (facts about no specific person)
    {AGENT}/Profiles             <- one note per profiled person + a "You" note
    {AGENT}/Actions              <- one note per action memory

A memory is one PARAGRAPH (a block of text between blank lines). There are NO
visible id tags: a memory's id is an 8-char hash derived from its note + text,
recomputed on every load. That way the notes stay clean prose you can hand-edit
freely — add, reword, reorder, or delete paragraphs and the index just follows.
A paragraph may optionally start with an "@Location:" prefix:

    General                       <- <h1> title (Notes derives it from line 1)

    Thomas uses iCloud for email, calendar, and notes.

    @Singapore: Gregor Gregersen founded Silver Bullion Group.

Profile notes carry an optional header (Aliases: / About:) before the memories.

Reads are synchronous (pure RAM). Writes update RAM + SQLite synchronously and
enqueue an async Notes rewrite (the affected note is regenerated wholesale from
RAM — idempotent). A background poll reloads notes whose modificationDate
advanced, so edits made by hand in Notes flow back in.
"""

import asyncio
import hashlib
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta

from loguru import logger

# Optional "@Location: text" prefix on a memory paragraph (multi-word location).
_LOC_PREFIX = re.compile(r"^@([^:@\n]{1,60}):\s+(.*)$", re.S)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _norm(content: str) -> str:
    """Whitespace/case-insensitive normal form for hashing + dedup."""
    return " ".join(str(content).split()).lower()


def _mem_id(note_key: str, content: str) -> str:
    """8-char combo id: hash of (which note) + (normalized paragraph text).

    Deriving it from content means it is never written into the note and is
    stable across reloads; folding in the note key keeps identical text in two
    different notes distinct.
    """
    return hashlib.sha1(f"{note_key}\x00{_norm(content)}".encode("utf-8")).hexdigest()[:8]


class NotesMemoryStore:
    def __init__(self, agent_name: str, stats_db_path: str, call_tool):
        """call_tool: async (toolset_key, tool_name, args) -> str text output."""
        self._agent = agent_name.strip() or "Agent"
        self._root = self._agent
        self._profiles = f"{self._agent}/Profiles"
        self._actions = f"{self._agent}/Actions"
        self._general_title = "General"
        self._call = call_tool

        self._db = sqlite3.connect(stats_db_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory_stats ("
            "memory_id TEXT PRIMARY KEY, recall_count INTEGER DEFAULT 0, "
            "last_recalled_at TEXT, created_at TEXT)"
        )
        self._db.execute("CREATE TABLE IF NOT EXISTS recall_log (memory_id TEXT, at TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

        # RAM index
        self._lock = threading.RLock()
        self._mem: dict[str, dict] = {}      # id -> {id, content, kind, person, location, note_path, order, created_at}
        self._people: dict[str, dict] = {}   # lower-name -> {name, aliases, description}
        self._note_mod: dict[str, str] = {}  # "folder\x00title" -> modificationDate

        # async write plumbing
        self._dirty: set[str] = set()         # note paths ("folder\x00title") to rewrite
        self._deleted_notes: set[str] = set() # notes to delete entirely
        self._dirty_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---- MCP helpers ----------------------------------------------------

    async def _mcp(self, tool: str, args: dict) -> str:
        return await self._call("notes", tool, args)

    async def _ensure_folders(self):
        for f in (self._root, self._profiles, self._actions):
            try:
                await self._mcp("create_folder", {"name": f})
            except Exception as exc:  # noqa: BLE001 — already-exists is fine
                logger.debug(f"notes folder {f}: {exc}")

    # ---- parsing / rendering -------------------------------------------

    @staticmethod
    def _note_key(folder: str, title: str) -> str:
        return f"{folder}\x00{title}"

    def _parse_note(self, folder: str, title: str, body: str, kind: str, person: str | None):
        """Parse one note's body into memory records + (for profiles) header.

        get_note returns the body as HTML (<div>line</div>...). Normalize tags
        to newlines, then group the lines into paragraphs (blank line = break);
        each paragraph is one memory. Header lines (title echo, Aliases:, About:)
        are recognized before the first memory.
        """
        import html
        text = re.sub(r"</div>|<br\s*/?>|</p>|</h[1-6]>", "\n", body, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)

        note_key = self._note_key(folder, title)
        aliases: list[str] = []
        description = ""
        memories: list[dict] = []
        seen_content = False
        para: list[str] = []

        def flush_para():
            nonlocal seen_content
            if not para:
                return
            content = " ".join(para).strip()
            para.clear()
            if not content:
                return
            location = None
            m = _LOC_PREFIX.match(content)
            if m:
                location, content = m.group(1).strip(), m.group(2).strip()
            if not content:
                return
            seen_content = True
            memories.append({
                "id": _mem_id(note_key, content), "content": content, "kind": kind,
                "person": person, "location": location, "note_path": note_key,
                "order": len(memories),
            })

        for raw in text.split("\n"):
            line = raw.strip()
            if not line:
                flush_para()
                continue
            if not seen_content and not para:
                # header region: title echo + Aliases:/About: (profiles)
                if line == title:
                    continue
                low = line.lower()
                if low.startswith("aliases:"):
                    aliases = [a.strip() for a in line[8:].split(",") if a.strip()]
                    continue
                if low.startswith("about:"):
                    description = line[6:].strip()
                    continue
            para.append(line)
        flush_para()
        return memories, aliases, description

    def _render_note(self, folder: str, title: str, include_title: bool) -> str:
        """Rebuild a note's HTML body from the current RAM records.

        The two Notes write ops treat the title differently, so callers must
        say which they're feeding:
          - update_note REPLACES the whole note, and Notes derives the title
            from line 1 -> include_title=True (body must lead with the title).
          - create_note PREPENDS the title param itself -> include_title=False
            (adding it here would duplicate the header).
        Memories are plain prose paragraphs, one blank line apart.
        """
        key = self._note_key(folder, title)
        parts: list[str] = []
        if include_title:
            # <h1> -> Notes "Heading" style; line 1 also becomes the note title.
            parts.append(f"<h1>{_esc(title)}</h1>")
        if folder == self._profiles:
            person = self._people.get(title.lower())
            if person:
                if person.get("aliases"):
                    parts.append(f"<div>Aliases: {_esc(', '.join(person['aliases']))}</div>")
                if person.get("description"):
                    parts.append(f"<div>About: {_esc(person['description'])}</div>")
        blank = "<div><br></div>"
        recs = sorted((r for r in self._mem.values() if r["note_path"] == key),
                      key=lambda r: (r.get("order", 0), r["id"]))
        for rec in recs:
            # A title always precedes the memories (included here, or prepended
            # by create_note), so every memory gets a blank line above it.
            parts.append(blank)
            loc = rec.get("location")
            content = f"@{loc}: {rec['content']}" if loc else rec["content"]
            parts.append(f"<div>{_esc(content)}</div>")
        return "\n".join(parts)

    # ---- stats ----------------------------------------------------------

    def _ensure_stats(self, mid: str, created_at: str):
        self._db.execute(
            "INSERT OR IGNORE INTO memory_stats (memory_id, created_at) VALUES (?, ?)",
            (mid, created_at),
        )

    # ---- load / refresh (async) ----------------------------------------

    async def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._dirty_event = asyncio.Event()
        await self._ensure_folders()
        await self.reload()

    async def reload(self):
        """Full rebuild of the RAM index from Notes."""
        mem: dict[str, dict] = {}
        people: dict[str, dict] = {}
        note_mod: dict[str, str] = {}
        import json

        async def load_folder(folder: str, kind: str, person_from_title: bool):
            try:
                listing = json.loads(await self._mcp("list_notes", {"folder": folder}))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"notes list {folder}: {exc}")
                return
            for meta in listing if isinstance(listing, list) else []:
                title = meta.get("title", "")
                note_mod[self._note_key(folder, title)] = meta.get("modificationDate", "")
                try:
                    note = json.loads(await self._mcp("get_note", {"title": title, "folder": folder}))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"notes get {folder}/{title}: {exc}")
                    continue
                person = title if person_from_title and title.lower() != "you" else None
                mems, aliases, desc = self._parse_note(folder, title, note.get("body", ""), kind, person)
                if person_from_title:
                    people[title.lower()] = {"name": title, "aliases": aliases, "description": desc}
                for rec in mems:
                    mem[rec["id"]] = rec

        await load_folder(self._profiles, "fact", True)
        await load_folder(self._actions, "action", False)
        # the General note lives directly in the agent root folder
        await load_folder(self._root, "fact", False)

        # backfill stats rows for any ids present in notes but not the sidecar,
        # and stamp each record with its (stable) created_at for recency sorting
        with self._lock:
            now = _now()
            for mid, rec in mem.items():
                self._ensure_stats(mid, now)
            self._db.commit()
            for mid, rec in mem.items():
                row = self._db.execute(
                    "SELECT created_at FROM memory_stats WHERE memory_id = ?", (mid,)
                ).fetchone()
                rec["created_at"] = row[0] if row and row[0] else now
            self._mem, self._people, self._note_mod = mem, people, note_mod
        logger.info(f"Notes memory loaded: {len(mem)} memories, {len(people)} people")

    async def refresh(self):
        """Poll for externally-edited notes and reload if any changed."""
        import json
        changed = False
        for folder in (self._profiles, self._actions, self._root):
            try:
                listing = json.loads(await self._mcp("list_notes", {"folder": folder}))
            except Exception:  # noqa: BLE001
                continue
            seen = set()
            for meta in listing if isinstance(listing, list) else []:
                k = self._note_key(folder, meta.get("title", ""))
                seen.add(k)
                if self._note_mod.get(k) != meta.get("modificationDate", ""):
                    changed = True
            # a note that vanished also counts as a change
            for k in list(self._note_mod):
                if k.startswith(folder + "\x00") and k not in seen:
                    changed = True
        if changed:
            logger.info("Notes changed externally — reloading memory index")
            await self.reload()

    # ---- async write drainer -------------------------------------------

    def _mark_dirty(self, note_path: str):
        self._dirty.add(note_path)
        if self._loop and self._dirty_event:
            self._loop.call_soon_threadsafe(self._dirty_event.set)

    async def run_writer(self):
        """Background task: flush dirty notes to Apple Notes as they appear."""
        assert self._dirty_event is not None
        while True:
            await self._dirty_event.wait()
            self._dirty_event.clear()
            await self.flush()

    async def flush(self):
        """Drain all pending note writes/deletes, awaiting completion."""
        while True:
            with self._lock:
                dirty = self._dirty; self._dirty = set()
                deleted = self._deleted_notes; self._deleted_notes = set()
            if not dirty and not deleted:
                return
            for key in deleted:
                folder, title = key.split("\x00", 1)
                try:
                    await self._mcp("delete_note", {"title": title, "folder": folder})
                    self._note_mod.pop(key, None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"notes delete {key}: {exc}")
            for key in dirty:
                if key in deleted:
                    continue
                folder, title = key.split("\x00", 1)
                try:
                    # update_note REPLACES the note and derives the title from
                    # line 1, so its body carries the <h1> title. Try that first;
                    # if the note doesn't exist yet, create a stub (create_note
                    # prepends the title itself -> body without it) and then
                    # update it so brand-new notes get the same header + layout.
                    header_body = self._render_note(folder, title, include_title=True)
                    try:
                        await self._mcp("update_note", {
                            "title": title, "folder": folder, "body": header_body,
                        })
                    except Exception:  # noqa: BLE001 — not found -> create then style
                        await self._mcp("create_note", {
                            "title": title, "folder": folder,
                            "body": self._render_note(folder, title, include_title=False),
                        })
                        await self._mcp("update_note", {
                            "title": title, "folder": folder, "body": header_body,
                        })
                    self._note_mod[key] = _now()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"notes write {key}: {exc}")

    # ---- routing --------------------------------------------------------

    def _route(self, kind: str, person: str | None) -> tuple[str, str]:
        """Which (folder, title) note a memory belongs in."""
        if kind == "action":
            return self._actions, None  # title assigned by caller (per-action note)
        if person:
            return self._profiles, person
        return self._root, self._general_title

    def _next_order(self, note_key: str) -> int:
        cur = [r.get("order", 0) for r in self._mem.values() if r["note_path"] == note_key]
        return (max(cur) + 1) if cur else 0

    # ---- read API (sync, from RAM) -------------------------------------

    def recall(self, keywords, person=None, location=None, limit=8, kind=None) -> dict:
        kw = [str(k) for k in (keywords or [])]
        terms = [w.lower() for kwd in kw for w in re.findall(r"[\w']+", kwd)]
        stems = {t[:4] for t in terms if len(t) > 3}
        with self._lock:
            hits = []
            for rec in self._mem.values():
                if kind and rec["kind"] != kind:
                    continue
                if person and (rec.get("person") or "").lower() != person.lower():
                    continue
                if location and (rec.get("location") or "").lower() != location.lower():
                    continue
                hay = f"{rec['content']} {rec.get('person') or ''} {rec.get('location') or ''}".lower()
                hay_stems = {w[:4] for w in re.findall(r"[\w']+", hay) if len(w) > 3}
                score = len(stems & hay_stems)
                if score:
                    hits.append((score, rec))
            hits.sort(key=lambda p: (p[0], p[1].get("created_at", "")), reverse=True)
            out = [self._public(r) for _s, r in hits[:limit]]
            if out:
                now = _now()
                self._db.executemany(
                    "INSERT INTO memory_stats (memory_id, recall_count, last_recalled_at) "
                    "VALUES (?, 1, ?) ON CONFLICT(memory_id) DO UPDATE SET "
                    "recall_count = recall_count + 1, last_recalled_at = excluded.last_recalled_at",
                    [(m["id"], now) for m in out],
                )
                self._db.executemany(
                    "INSERT INTO recall_log (memory_id, at) VALUES (?, ?)",
                    [(m["id"], now) for m in out],
                )
                self._db.commit()
        return {"memories": out, **({"person": person} if person else {})}

    def _public(self, rec: dict) -> dict:
        return {
            "id": rec["id"], "content": rec["content"], "context": "",
            "person": rec.get("person"), "location": rec.get("location"),
            "kind": rec["kind"], "created_at": rec.get("created_at") or _now(),
        }

    def recent(self, limit=5, kind=None) -> list[dict]:
        with self._lock:
            recs = [r for r in self._mem.values() if not kind or r["kind"] == kind]
        pub = [self._public(r) for r in recs]
        pub.sort(key=lambda m: m["created_at"], reverse=True)
        return pub[:limit]

    def top_recalled(self, limit=5, exclude_ids=None, window_days=30) -> list[dict]:
        exclude = exclude_ids or set()
        cutoff = (datetime.now().astimezone() - timedelta(days=window_days)).isoformat(timespec="seconds")
        rows = self._db.execute(
            "SELECT memory_id, COUNT(*) c FROM recall_log WHERE at >= ? "
            "GROUP BY memory_id ORDER BY c DESC, MAX(at) DESC LIMIT ?",
            (cutoff, limit + len(exclude)),
        ).fetchall()
        out = []
        with self._lock:
            for mid, cnt in rows:
                if mid in exclude or mid not in self._mem:
                    continue
                m = self._public(self._mem[mid]); m["recall_count"] = cnt
                out.append(m)
        return out[:limit]

    def list_people(self) -> list[dict]:
        with self._lock:
            people = []
            for p in self._people.values():
                if p["name"].lower() == "you":
                    continue
                count = sum(1 for r in self._mem.values() if (r.get("person") or "").lower() == p["name"].lower())
                people.append({"name": p["name"], "description": p.get("description", ""),
                               "aliases": p.get("aliases", []), "since": _now()[:10], "memories": count})
        return people

    def resolve_person(self, name: str) -> dict:
        n = name.strip().lower().lstrip("@")
        with self._lock:
            for p in self._people.values():
                if p["name"].lower() == n:
                    return {"person": p["name"]}
            alias_hits = [p["name"] for p in self._people.values()
                          if n in [a.lower().lstrip("@") for a in p.get("aliases", [])]]
            if len(alias_hits) == 1:
                return {"person": alias_hits[0]}
            if len(alias_hits) > 1:
                return {"candidates": [{"name": x} for x in alias_hits]}
            part = [p["name"] for p in self._people.values() if n in p["name"].lower()]
            if len(part) == 1:
                return {"person": part[0]}
            if len(part) > 1:
                return {"candidates": [{"name": x} for x in part]}
            return {"candidates": []}  # truly unknown -> caller auto-registers

    # ---- write API (sync: RAM + SQLite now, Notes async) ---------------

    def add_person(self, name: str, description: str = "", aliases=None) -> dict:
        name = name.strip()
        if not name:
            return {"error": "person needs a name"}
        aliases = [a.strip() for a in (aliases or []) if a and a.strip()]
        with self._lock:
            key = name.lower()
            existing = self._people.get(key)
            if existing:
                merged = list(dict.fromkeys(existing.get("aliases", []) + aliases))
                existing.update({"description": description or existing.get("description", ""), "aliases": merged})
                self._mark_dirty(self._note_key(self._profiles, existing["name"]))
                return {"person": existing["name"], "updated": True, "aliases": merged}
            self._people[key] = {"name": name, "aliases": aliases, "description": description}
            self._mark_dirty(self._note_key(self._profiles, name))
        return {"person": name, "registered": True, "aliases": aliases}

    def meta_get(self, key: str):
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, key: str, value: str):
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self._db.commit()

    def _put(self, rec: dict):
        """Insert/replace a record under its content-derived id + ensure stats."""
        self._mem[rec["id"]] = rec
        self._ensure_stats(rec["id"], rec.get("created_at") or _now())
        self._db.commit()

    def remember(self, content, context="", memory_id=None, location=None, person=None,
                 kind=None, created_at=None, skip_dedup=False) -> dict:
        content = str(content).strip()
        if not content:
            return {"error": "nothing to remember"}
        if kind not in (None, "fact", "action"):
            return {"error": f"invalid kind {kind!r}"}
        resolved = None
        if person:
            r = self.resolve_person(person)
            if "person" in r:
                resolved = r["person"]
            elif r.get("candidates"):
                return {"error": f"person {person!r} is not uniquely known",
                        "candidates": [c["name"] for c in r["candidates"]]}
            else:
                self.add_person(person)
                resolved = person.strip()
        with self._lock:
            if memory_id is not None:
                mid = str(memory_id)
                rec = self._mem.get(mid)
                if not rec:
                    return {"error": f"no memory with id {memory_id}"}
                old_note = rec["note_path"]
                new_kind = kind or rec["kind"]
                new_person = resolved if person is not None else rec.get("person")
                folder, title = self._route(new_kind, new_person)
                if folder == self._actions and title is None:
                    title = _slug(content)
                new_note = self._note_key(folder, title)
                # content and/or note changed -> the id is re-derived
                new_id = _mem_id(new_note, content)
                rec = {**rec, "id": new_id, "content": content, "kind": new_kind,
                       "person": new_person, "note_path": new_note}
                if location is not None:
                    rec["location"] = location or None
                if new_note != old_note:
                    rec["order"] = self._next_order(new_note)
                # re-key RAM + carry stats across the id change
                self._mem.pop(mid, None)
                self._mem[new_id] = rec
                if new_id != mid:
                    self._migrate_stats(mid, new_id)
                self._mark_dirty(new_note)
                if old_note != new_note:
                    self._mark_dirty(old_note)
                return {"id": new_id, "edited": True, "person": rec.get("person"),
                        "location": rec.get("location"), "created_at": rec.get("created_at")}
            # dedup guard
            if not skip_dedup:
                dup = self._find_similar(content, kind or "fact")
                if dup is not None:
                    return {"error": f"very similar to existing memory {dup} — pass id={dup} to update it",
                            "similar_id": dup}
            k = kind or "fact"
            folder, title = self._route(k, resolved)
            if folder == self._actions and title is None:
                title = _slug(content)
            note_key = self._note_key(folder, title)
            mid = _mem_id(note_key, content)
            rec = {"id": mid, "content": content, "kind": k, "person": resolved,
                   "location": (location or None), "note_path": note_key,
                   "order": self._next_order(note_key), "created_at": created_at or _now()}
            self._put(rec)
            self._mark_dirty(note_key)
        return {"id": mid, "created_at": rec["created_at"], "person": resolved, "location": location or None}

    def _migrate_stats(self, old_id: str, new_id: str):
        """Carry recall stats from an old (pre-edit) id to the new one."""
        self._ensure_stats(new_id, _now())
        self._db.execute(
            "UPDATE memory_stats SET recall_count = recall_count + "
            "COALESCE((SELECT recall_count FROM memory_stats WHERE memory_id = ?), 0), "
            "last_recalled_at = COALESCE((SELECT last_recalled_at FROM memory_stats WHERE memory_id = ?), last_recalled_at) "
            "WHERE memory_id = ?", (old_id, old_id, new_id),
        )
        self._db.execute("UPDATE recall_log SET memory_id = ? WHERE memory_id = ?", (new_id, old_id))
        self._db.execute("DELETE FROM memory_stats WHERE memory_id = ?", (old_id,))
        self._db.commit()

    def _find_similar(self, content: str, kind: str):
        stems = {w[:4].lower() for w in re.findall(r"[\w']+", content) if len(w) > 3}
        if len(stems) < 4:
            return None
        for rec in self._mem.values():
            if rec["kind"] != kind:
                continue
            other = {w[:4].lower() for w in re.findall(r"[\w']+", rec["content"]) if len(w) > 3}
            if other and len(stems & other) / min(len(stems), len(other)) >= 0.6:
                return rec["id"]
        return None

    def forget(self, memory_id) -> dict:
        mid = str(memory_id)
        with self._lock:
            rec = self._mem.pop(mid, None)
            if not rec:
                return {"error": f"no memory with id {memory_id}"}
            self._db.execute("DELETE FROM memory_stats WHERE memory_id = ?", (mid,))
            self._db.execute("DELETE FROM recall_log WHERE memory_id = ?", (mid,))
            self._db.commit()
            # Forgetting a memory never deletes the note — just re-render it
            # (an emptied note keeps its title/header). Whole-note deletion is
            # reserved for removing a profile (person).
            self._mark_dirty(rec["note_path"])
        return {"id": mid, "forgotten": rec["content"]}

    def seed_actions(self, seeds: list[str]) -> int:
        added = 0
        with self._lock:
            existing = [r["content"][:40] for r in self._mem.values() if r["kind"] == "action"]
        for seed in seeds:
            if any(seed[:40] == e for e in existing):
                continue
            self.remember(seed, kind="action")
            added += 1
        return added

    def edit_person(self, person, new_name=None, description=None, aliases=None) -> dict:
        r = self.resolve_person(person)
        if "person" not in r:
            return {"error": f"person {person!r} is not uniquely known",
                    "candidates": [c["name"] for c in r.get("candidates", [])]}
        with self._lock:
            rec = self._people[r["person"].lower()]
            old_name = rec["name"]
            if aliases:
                rec["aliases"] = list(dict.fromkeys(rec.get("aliases", []) + [a.strip() for a in aliases]))
            if description is not None:
                rec["description"] = description
            if new_name and new_name.strip() and new_name.strip().lower() != old_name.lower():
                new = new_name.strip()
                rec["aliases"] = list(dict.fromkeys(rec.get("aliases", []) + [old_name]))
                rec["name"] = new
                self._people[new.lower()] = self._people.pop(old_name.lower())
                new_note = self._note_key(self._profiles, new)
                # the note key changed, so every memory's id is re-derived
                for old_id in [m["id"] for m in self._mem.values()
                               if (m.get("person") or "").lower() == old_name.lower()]:
                    m = self._mem.pop(old_id)
                    m["person"] = new
                    m["note_path"] = new_note
                    m["id"] = _mem_id(new_note, m["content"])
                    self._mem[m["id"]] = m
                    if m["id"] != old_id:
                        self._migrate_stats(old_id, m["id"])
                self._deleted_notes.add(self._note_key(self._profiles, old_name))
                self._mark_dirty(new_note)
            else:
                self._mark_dirty(self._note_key(self._profiles, old_name))
        return {"person": rec["name"], "updated": True}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _slug(text: str) -> str:
    """Short note title for an action memory (its trigger phrase)."""
    head = re.split(r"[:.]", text, 1)[0].strip()
    return (head[:50] or text[:50]).strip()
