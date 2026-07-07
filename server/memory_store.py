"""SQLite-backed long-term memory for the voice agent.

Memories live in an FTS5 (full-text search) table so recall can match any of
the given keywords (OR semantics) and rank results by relevance. Each memory
carries a local-time timestamp, a snapshot of the conversation context that was
present when it was stored, and optional tags: a location and a person.

People are tracked in a segregated registry table. Tagging or filtering by
person goes through resolve_person(); ambiguous or unknown references return
candidates instead of guessing, so the agent can ask the user.
"""

import re
import sqlite3
import threading
from datetime import datetime

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class MemoryStore:
    """Tiny synchronous SQLite store; call via asyncio.to_thread from handlers."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._migrate()

    def _migrate(self):
        c = self._conn
        c.execute(
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE, "
            "description TEXT DEFAULT '', created_at TEXT)"
        )
        # aliases: JSON list of nicknames / online identifiers (added later)
        cols = {r[1] for r in c.execute("PRAGMA table_info(people)").fetchall()}
        if "aliases" not in cols:
            c.execute("ALTER TABLE people ADD COLUMN aliases TEXT DEFAULT '[]'")
        # memories is an FTS5 table; adding columns requires a rebuild.
        # The porter tokenizer stems terms so recall matches inflections
        # ("sisters" finds "sister", "visited" finds "visit"), not just
        # exact words. Changing the tokenizer requires a table rebuild.
        create_fts = (
            "CREATE VIRTUAL TABLE memories USING fts5("
            "content, context, location, person, created_at UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE name='memories' AND type='table'"
        ).fetchone()
        if row is None:
            c.execute(create_fts)
        elif "location" not in row[0]:
            # oldest schema: content, context, created_at — rebuild with tags
            old = c.execute("SELECT content, context, created_at FROM memories").fetchall()
            c.execute("DROP TABLE memories")
            c.execute(create_fts)
            c.executemany(
                "INSERT INTO memories (content, context, location, person, created_at) "
                "VALUES (?, ?, '', '', ?)",
                old,
            )
        elif "porter" not in row[0]:
            # right columns, pre-stemming tokenizer — rebuild in place,
            # keeping rowids stable (they serve as memory ids).
            old = c.execute(
                "SELECT rowid, content, context, location, person, created_at FROM memories"
            ).fetchall()
            c.execute("DROP TABLE memories")
            c.execute(create_fts)
            c.executemany(
                "INSERT INTO memories (rowid, content, context, location, person, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                old,
            )
        c.commit()

    # --- people -----------------------------------------------------------

    @staticmethod
    def _norm_alias(alias: str) -> str:
        """Normalize an identifier: strip whitespace and a leading @."""
        return alias.strip().lstrip("@").lower()

    def add_person(self, name: str, description: str = "",
                   aliases: list[str] | None = None) -> dict:
        import json

        name = name.strip()
        if not name:
            return {"error": "person needs a name"}
        new_aliases = [a.strip() for a in (aliases or []) if a and a.strip()]
        with _LOCK:
            existing = self._conn.execute(
                "SELECT id, name, description, aliases FROM people "
                "WHERE name = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            if existing:
                merged = list(json.loads(existing[3] or "[]"))
                norms = {self._norm_alias(a) for a in merged}
                merged += [a for a in new_aliases if self._norm_alias(a) not in norms]
                self._conn.execute(
                    "UPDATE people SET description = ?, aliases = ? WHERE id = ?",
                    (description or existing[2], json.dumps(merged), existing[0]),
                )
                self._conn.commit()
                return {"person": existing[1], "updated": True,
                        "description": description or existing[2], "aliases": merged}
            self._conn.execute(
                "INSERT INTO people (name, description, aliases, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, description, json.dumps(new_aliases), _now()),
            )
            self._conn.commit()
        return {"person": name, "registered": True, "aliases": new_aliases}

    def edit_person(self, reference: str, new_name: str | None = None,
                    description: str | None = None,
                    aliases: list[str] | None = None) -> dict:
        """Correct or extend a registered person: rename (memories follow,
        the old spelling stays reachable as an alias), update the
        description, and/or add aliases."""
        import json

        r = self.resolve_person(reference)
        if "person" not in r:
            return {
                "error": f"person {reference!r} is not uniquely known",
                "candidates": [c["name"] for c in r.get("candidates", [])],
            }
        with _LOCK:
            row = self._conn.execute(
                "SELECT id, name, description, aliases FROM people "
                "WHERE name = ? COLLATE NOCASE",
                (r["person"],),
            ).fetchone()
            pid, name, desc, raw = row
            merged = list(json.loads(raw or "[]"))
            norms = {self._norm_alias(a) for a in merged}
            for a in aliases or []:
                if a and a.strip() and self._norm_alias(a) not in norms:
                    merged.append(a.strip())
                    norms.add(self._norm_alias(a))
            final_name = (new_name or name).strip() or name
            renamed = final_name.lower() != name.lower()
            if renamed:
                if self._norm_alias(name) not in norms:
                    merged.append(name)  # old spelling remains resolvable
                self._conn.execute(
                    "UPDATE memories SET person = ? WHERE person = ? COLLATE NOCASE",
                    (final_name, name),
                )
            self._conn.execute(
                "UPDATE people SET name = ?, description = ?, aliases = ? WHERE id = ?",
                (final_name, description if description is not None else desc,
                 json.dumps(merged), pid),
            )
            self._conn.commit()
        return {"person": final_name, "updated": True,
                "renamed_from": name if renamed else None,
                "description": description if description is not None else desc,
                "aliases": merged}

    def list_people(self) -> list[dict]:
        import json

        with _LOCK:
            rows = self._conn.execute(
                "SELECT p.name, p.description, p.aliases, p.created_at, "
                "(SELECT count(*) FROM memories m WHERE m.person = p.name COLLATE NOCASE) "
                "FROM people p ORDER BY p.name"
            ).fetchall()
        return [
            {"name": r[0], "description": r[1], "aliases": json.loads(r[2] or "[]"),
             "since": r[3], "memories": r[4]}
            for r in rows
        ]

    def resolve_person(self, reference: str) -> dict:
        """Resolve a name, alias (handle/email/nickname), or description to a
        registered person.

        Returns {"person": name} on a unique match, otherwise {"candidates":
        [...]} (possibly empty) so the caller can ask the user.
        """
        import json

        ref = reference.strip()
        if not ref:
            return {"candidates": []}
        norm = self._norm_alias(ref)
        with _LOCK:
            exact = self._conn.execute(
                "SELECT name FROM people WHERE name = ? COLLATE NOCASE", (ref,)
            ).fetchone()
            if exact:
                return {"person": exact[0]}
            rows = self._conn.execute(
                "SELECT name, description, aliases FROM people"
            ).fetchall()
        # exact alias match (handles @handle, email, nickname)
        alias_hits = [
            r for r in rows
            if norm in {self._norm_alias(a) for a in json.loads(r[2] or "[]")}
        ]
        if len(alias_hits) == 1:
            return {"person": alias_hits[0][0]}
        if alias_hits:
            rows = alias_hits
        else:
            # fuzzy: substring across name, description, aliases
            rows = [
                r for r in rows
                if norm in r[0].lower()
                or norm in (r[1] or "").lower()
                or any(norm in self._norm_alias(a) for a in json.loads(r[2] or "[]"))
            ]
        if len(rows) == 1:
            return {"person": rows[0][0]}
        return {
            "candidates": [
                {"name": n, "description": d, "aliases": json.loads(a or "[]")}
                for n, d, a in rows
            ]
        }

    # --- memories ---------------------------------------------------------

    def remember(
        self,
        content: str,
        context: str,
        memory_id: int | None = None,
        location: str | None = None,
        person: str | None = None,
    ) -> dict:
        """Insert or overwrite (memory_id) a memory, optionally tagged."""
        resolved = ""
        person_registered = False
        if person:
            r = self.resolve_person(person)
            if "person" in r:
                resolved = r["person"]
            elif r["candidates"]:
                names = [c["name"] for c in r["candidates"]]
                return {
                    "error": f"person {person!r} is not uniquely known",
                    "candidates": names,
                    "hint": "ask the user which person is meant, or register them with add_person",
                }
            else:
                # Unknown but unambiguous: register on the fly rather than
                # bouncing the model through an error round-trip.
                self.add_person(person)
                resolved = person.strip()
                person_registered = True
        created_at = _now()
        with _LOCK:
            if memory_id is not None:
                cur = self._conn.execute(
                    "UPDATE memories SET content = ?, context = ?, location = ?, "
                    "person = ?, created_at = ? WHERE rowid = ?",
                    (content, context, location or "", resolved, created_at, memory_id),
                )
                self._conn.commit()
                if cur.rowcount == 0:
                    return {"error": f"no memory with id {memory_id}"}
                out = {"id": memory_id, "created_at": created_at, "edited": True,
                       "person": resolved or None, "location": location or None}
                if person_registered:
                    out["person_registered"] = True
                return out
            cur = self._conn.execute(
                "INSERT INTO memories (content, context, location, person, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (content, context, location or "", resolved, created_at),
            )
            self._conn.commit()
        out = {"id": cur.lastrowid, "created_at": created_at,
               "person": resolved or None, "location": location or None}
        if person_registered:
            out["person_registered"] = True
        return out

    def recall(
        self,
        keywords: list[str],
        person: str | None = None,
        location: str | None = None,
        limit: int = 8,
    ) -> dict:
        """Search memories: keywords OR-matched, optional person/location filters."""
        def words(kw: str) -> list[str]:
            return re.findall(r"[\w']+", str(kw))

        # Each multi-word keyword contributes its phrase (ranking boost) plus
        # its individual words, all OR-ed, so partial matches still hit.
        kw_terms: list[str] = []
        for kw in keywords:
            ws = words(kw)
            if not ws:
                continue
            if len(ws) > 1:
                kw_terms.append('"' + " ".join(ws) + '"')
            kw_terms.extend(f'"{w}"' for w in ws)
        kw_terms = list(dict.fromkeys(kw_terms))

        clauses = []
        if kw_terms:
            clauses.append("(" + " OR ".join(kw_terms) + ")")
        resolved = None
        if person:
            r = self.resolve_person(person)
            if "person" not in r:
                names = [c["name"] for c in r["candidates"]]
                return {
                    "error": f"person {person!r} is not uniquely known",
                    "candidates": names,
                    "hint": "ask the user which person is meant",
                }
            resolved = r["person"]
            clauses.append('person:"' + re.sub(r'["\']', "", resolved).strip() + '"')
        if location:
            clauses.append('location:"' + re.sub(r'["\']', "", location).strip() + '"')
        if not clauses:
            return {"memories": []}
        query = " AND ".join(clauses)
        with _LOCK:
            rows = self._conn.execute(
                "SELECT rowid, content, context, location, person, created_at "
                "FROM memories WHERE memories MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            if not rows and kw_terms:
                # FTS missed entirely (odd inflection, partial word): fall back
                # to a substring scan over content, newest first.
                subs = list(dict.fromkeys(
                    w.lower() for kw in keywords for w in words(kw)
                ))
                where = " OR ".join(["lower(content) LIKE ?"] * len(subs))
                params: list = [f"%{s}%" for s in subs]
                if resolved:
                    where = f"({where}) AND person = ?"
                    params.append(resolved)
                rows = self._conn.execute(
                    "SELECT rowid, content, context, location, person, created_at "
                    f"FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
                    (*params, limit),
                ).fetchall()
        memories = [
            {
                "id": r[0], "content": r[1], "context": r[2],
                "location": r[3] or None, "person": r[4] or None, "created_at": r[5],
            }
            for r in rows
        ]
        return {"memories": memories, **({"person": resolved} if resolved else {})}

    def recent(self, limit: int = 5) -> list[dict]:
        """The most recently stored/edited memories, newest first."""
        with _LOCK:
            rows = self._conn.execute(
                "SELECT rowid, content, person, location, created_at FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "content": r[1], "person": r[2] or None,
             "location": r[3] or None, "created_at": r[4]}
            for r in rows
        ]

    def forget(self, memory_id: int) -> dict:
        with _LOCK:
            row = self._conn.execute(
                "SELECT content FROM memories WHERE rowid = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return {"error": f"no memory with id {memory_id}"}
            self._conn.execute("DELETE FROM memories WHERE rowid = ?", (memory_id,))
            self._conn.commit()
        return {"id": memory_id, "forgotten": row[0]}
