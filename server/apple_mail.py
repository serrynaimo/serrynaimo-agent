"""Native Apple Mail tools — a lean replacement for the mcp-apple-mail server.

Optimised for an AI that struggles with the MCP's sharp edges:

- **No account names.** Search spans every account and mailbox; read / reply /
  trash / mark act on a message directly, so the model never has to discover or
  spell an exact account display name (the MCP's biggest failure mode).
- **Stable cross-call ids.** Search mints a short handle (``m1``, ``m2`` …) for
  each email, backed by its globally-unique RFC822 ``message id``. The model
  re-references an email it already found by that handle — no re-search, no
  brittle subject-substring matching. Handles live for the process, so they keep
  working across turns and after context summarisation.
- **Broad-net search.** Every query word is OR-matched across subject *and*
  sender, newest first, capped at 40 with offset pagination — one call usually
  finds it.
- **Compact results.** Search returns metadata only (no bodies); the full,
  cleaned text of one email comes from ``read`` (paged). No raw-source tool.
- **Trash without an account.** ``delete`` moves a message to its account's
  Trash natively, so cleanup never needs the account spelled out.

Everything runs through a single ``osascript`` per call. Fields are joined with
ASCII unit/record separators (never present in mail text), so no delimiter can
be corrupted by a subject or address.
"""

import asyncio
from collections import OrderedDict
from datetime import datetime

from loguru import logger

import fuzzy

# Field / record separators — ASCII unit + record separator, never in mail text.
_US, _RS = "\x1f", "\x1e"

# Per-call osascript timeout (seconds). Reading one body can be slow on IMAP.
_TIMEOUT = 90
# Max characters of body per read() page — keeps a single tool result small.
_PAGE = 4000
# Newest-first result cap for search.
_MAX_RESULTS = 40
# Safety cap on how many matches we pull per mailbox before sorting.
_PER_MAILBOX_CAP = 150
# How many resolved handles to remember.
_CACHE_MAX = 400

# System folders we never search (noise + slow); default case-insensitive match.
_SKIP_MAILBOXES = {
    "trash", "deleted messages", "deleted items", "bin",
    "junk", "junk e-mail", "spam",
    "sent", "sent messages", "sent mail", "sent items",
    "drafts", "outbox",
}


def _osa(s) -> str:
    """Escape a Python value for an AppleScript double-quoted literal.

    Folds curly quotes to ASCII (so keywords match smart-punctuation subjects),
    then escapes backslash/quote and turns newlines/tabs into AppleScript's own
    \\n / \\t escapes (which it does interpret inside string literals).
    """
    s = str(s)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n").replace("\t", "\\t")
    return s


# AppleScript helpers appended to every script: pad, ISO date, field cleaner,
# short preview, and a message finder that locates a message anywhere by a
# caller-supplied `whose` clause (RFC822 id or numeric id), trying one account
# first for speed then falling back to all accounts.
_PRELUDE = f'''
on p2(n)
	set s to (n as integer) as text
	if (length of s) < 2 then set s to "0" & s
	return s
end p2
on isoDate(d)
	return ((year of d) as text) & "-" & p2(month of d as integer) & "-" & p2(day of d) & "T" & p2(hours of d) & ":" & p2(minutes of d) & ":" & p2(seconds of d)
end isoDate
on clean(t)
	set t to t as text
	set AppleScript's text item delimiters to {{return, linefeed, tab, "{_US}", "{_RS}"}}
	set parts to text items of t
	set AppleScript's text item delimiters to " "
	set t to parts as string
	set AppleScript's text item delimiters to ""
	return t
end clean
'''


def _find_handler(clause: str, account: str) -> str:
    """AppleScript `findMsg()` handler that returns the message matching `clause`.

    Tries the stored account's mailboxes first (fast), then every account.
    Returns `missing value` if nothing matches.
    """
    acc = _osa(account)
    return f'''
on findMsg()
	tell application "Mail"
		if "{acc}" is not "" then
			try
				set a to account "{acc}"
				repeat with mb in (every mailbox of a)
					try
						set hits to (messages of mb whose {clause})
						if (count of hits) > 0 then return item 1 of hits
					end try
				end repeat
			end try
		end if
		repeat with a in every account
			repeat with mb in (every mailbox of a)
				try
					set hits to (messages of mb whose {clause})
					if (count of hits) > 0 then return item 1 of hits
				end try
			end repeat
		end repeat
		return missing value
	end tell
end findMsg
'''


def _when(iso: str) -> str:
    """Spoken-friendly 'when' for an ISO datetime, e.g. 'today', '3 days ago'."""
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return ""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    days = (now.date() - dt.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        w = days // 7
        return f"{w} week{'s' if w != 1 else ''} ago"
    if days < 365:
        m = days // 30
        return f"{m} month{'s' if m != 1 else ''} ago"
    y = days // 365
    return f"{y} year{'s' if y != 1 else ''} ago"


class AppleMail:
    """Native Apple Mail operations with short, stable cross-call handles."""

    def __init__(self):
        self._counter = 0
        self._by_id: OrderedDict[str, dict] = OrderedDict()  # handle -> entry
        self._by_key: dict[str, str] = {}                    # rfc/num key -> handle
        self._draft_counter = 0
        self._drafts: OrderedDict[str, dict] = OrderedDict()  # "d1" -> draft info

    # ---- osascript -------------------------------------------------------

    @staticmethod
    async def _run(script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError("Mail took too long to respond")
        if proc.returncode != 0:
            msg = (err or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(msg or "AppleScript failed")
        return out.decode("utf-8", "replace")

    # ---- handle cache ----------------------------------------------------

    def _handle_for(self, rfc: str, numeric: str, account: str, meta: dict) -> str:
        """Return a stable short handle for an email, minting one if new."""
        key = rfc.strip() or f"num:{numeric}:{account}"
        sid = self._by_key.get(key)
        if sid is None:
            self._counter += 1
            sid = f"m{self._counter}"
            self._by_key[key] = sid
        clause = (
            f'message id is "{_osa(rfc)}"' if rfc.strip()
            else f"id is {int(numeric)}" if numeric.isdigit()
            else None
        )
        entry = {"clause": clause, "account": account, **meta}
        self._by_id[sid] = entry
        self._by_id.move_to_end(sid)
        while len(self._by_id) > _CACHE_MAX:
            old_sid, _ = self._by_id.popitem(last=False)
            self._by_key = {k: v for k, v in self._by_key.items() if v != old_sid}
        return sid

    def _resolve(self, sid: str) -> dict | None:
        entry = self._by_id.get((sid or "").strip())
        if entry and entry.get("clause"):
            self._by_id.move_to_end(sid.strip())
        return entry if (entry and entry.get("clause")) else None

    @staticmethod
    def _unknown_handle(sid: str) -> dict:
        return {"error": f"unknown email id '{sid}' — run search_email first, then "
                         "use the id it returns"}

    # ---- search ----------------------------------------------------------

    @staticmethod
    def _date_setup(date_from: str, date_to: str) -> tuple[str, list[str], str | None]:
        """AppleScript lines defining fromDate/toDate + the matching conditions.

        Dates are built by mutating `current date` (locale-proof — AppleScript's
        `date "..."` literal parsing depends on system locale). Returns
        (setup_lines, conditions, error)."""
        setup, conds = [], []
        for label, var, raw in (("date_from", "fromDate", date_from),
                                ("date_to", "toDate", date_to)):
            raw = str(raw or "").strip()
            if not raw:
                continue
            try:
                d = datetime.strptime(raw[:10], "%Y-%m-%d")
            except ValueError:
                return "", [], f"{label} must be YYYY-MM-DD, got {raw!r}"
            secs = 0 if var == "fromDate" else 86399  # from midnight / to 23:59:59
            setup.append(
                f"set {var} to current date\n"
                f"set time of {var} to {secs}\n"
                f"set day of {var} to 1\n"
                f"set year of {var} to {d.year}\n"
                f"set month of {var} to {d.month}\n"
                f"set day of {var} to {d.day}"
            )
            conds.append(f"date received {'>=' if var == 'fromDate' else '<='} {var}")
        return "\n".join(setup), conds, None

    async def search(self, query: str, unread_only: bool = False,
                     days: int | None = None, offset: int = 0,
                     date_from: str = "", date_to: str = "",
                     folder: str = "") -> dict:
        terms = [t for t in str(query or "").split() if t]
        folder = str(folder or "").strip()
        base_conds: list[str] = []
        if unread_only:
            base_conds.append("read status is false")
        if days and days > 0:
            base_conds.append(f"date received > ((current date) - {int(days) * 86400})")
        date_setup, date_conds, date_err = self._date_setup(date_from, date_to)
        if date_err:
            return {"error": date_err}
        base_conds.extend(date_conds)
        inbox_only = not terms and not folder
        # Inbox mode with no filters returns the WHOLE inbox (newest first, paged)
        # with each email explicitly marked read/unread — no clever windowing.
        logger.info(f"search_email query={query!r} unread={unread_only} days={days} "
                    f"offset={offset} folder={folder!r} inbox_only={inbox_only}")

        def _where(term_words: list[str]) -> str:
            conds = list(base_conds)
            if term_words:
                ors = []
                for t in term_words:
                    e = _osa(t)
                    ors.append(f'subject contains "{e}"')
                    ors.append(f'sender contains "{e}"')
                conds.insert(0, "(" + " or ".join(ors) + ")")
            return f" whose {' and '.join(conds)}" if conds else ""

        # The per-message record emitted by both modes (same field order). The
        # read flag defaults to read — only an explicit false marks unread, so a
        # failed property fetch can never invent unread mail.
        emit = '''
						try
							set rfcId to ""
							try
								set rfcId to (message id of m) as text
							end try
							set readFlag to "1"
							try
								if read status of m is false then set readFlag to "0"
							end try
							set out to out & ((id of m) as text) & US & rfcId & US & (my clean(subject of m)) & US & (my clean(sender of m)) & US & (my clean(mbName)) & US & (my clean(accName)) & US & readFlag & US & (my isoDate(date received of m)) & RS
						end try'''

        def build_script(term_words: list[str]) -> str:
            where = _where(term_words)
            if inbox_only:
                # No keywords → each account's own Inbox mailbox. (NOT the unified
                # `inbox` object — its whose-filters and per-message properties are
                # unreliable, e.g. read status coming back wrong.)
                return f'''{_PRELUDE}
set US to "{_US}"
set RS to "{_RS}"
{date_setup}
tell application "Mail"
	set out to ""
	repeat with a in every account
		set accName to name of a
		set mbName to "Inbox"
		set ib to missing value
		try
			set ib to mailbox "INBOX" of a
		on error
			try
				set ib to mailbox "Inbox" of a
			end try
		end try
		if ib is not missing value then
			try
				set matches to (messages of ib{where})
				if (count of matches) > {_PER_MAILBOX_CAP} then set matches to items 1 thru {_PER_MAILBOX_CAP} of matches
				repeat with m in matches
{emit}
				end repeat
			end try
		end if
	end repeat
	return out
end tell'''
            # A named folder searches ONLY mailboxes matching that name (prefix,
            # case-insensitive — "sent" matches "Sent Messages"), skip list
            # bypassed: naming a folder means the user wants inside it. Otherwise
            # every mailbox except the system skip list.
            if folder:
                fl = _osa(folder)
                mb_cond = f'(mbName is "{fl}" or mbName starts with "{fl}")'
            else:
                mb_cond = "skipNames does not contain mbName"
            skip = "{" + ", ".join(f'"{_osa(n)}"' for n in sorted(_SKIP_MAILBOXES)) + "}"
            return f'''{_PRELUDE}
set US to "{_US}"
set RS to "{_RS}"
{date_setup}
set skipNames to {skip}
tell application "Mail"
	set out to ""
	repeat with a in every account
		set accName to name of a
		repeat with mb in (every mailbox of a)
			try
				set mbName to name of mb
			on error
				set mbName to ""
			end try
			if {mb_cond} then
				try
					set matches to (messages of mb{where})
					if (count of matches) > {_PER_MAILBOX_CAP} then set matches to items 1 thru {_PER_MAILBOX_CAP} of matches
					repeat with m in matches
{emit}
					end repeat
				end try
			end if
		end repeat
	end repeat
	return out
end tell'''

        async def scan(term_words: list[str]) -> list[dict]:
            raw = await self._run(build_script(term_words))
            rows = []
            seen = set()
            for rec in raw.split(_RS):
                if not rec.strip():
                    continue
                f = rec.split(_US)
                if len(f) < 8:
                    continue
                numeric, rfc, subject, sender, mailbox, account, read, iso = f[:8]
                dedup = rfc.strip() or f"{numeric}:{account}"
                if dedup in seen:
                    continue
                seen.add(dedup)
                rows.append({
                    "numeric": numeric, "rfc": rfc, "subject": subject.strip(),
                    "sender": sender.strip(), "mailbox": mailbox, "account": account,
                    "unread": read == "0", "iso": iso,  # AppleScript emits "1" = read
                })
            return rows

        try:
            rows = await scan(terms)
        except RuntimeError as exc:
            return {"error": f"could not search mail: {exc}"}

        # Fuzzy fallback: zero literal hits often just means a misspelled name
        # (dictation writes "Quantas" for Qantas). Retry once with substring
        # stems of the terms, then keep only candidates whose sender/subject
        # actually resembles what was asked — labelled approximate below.
        approximate = False
        if not rows and terms:
            stems: list[str] = []
            for t in terms:
                for v in fuzzy.variants(t):
                    if v not in stems:
                        stems.append(v)
            if stems:
                logger.info(f"search_email fuzzy retry with stems {stems}")
                try:
                    candidates = await scan(stems)
                except RuntimeError:
                    candidates = []
                rows = [r for r in candidates
                        if fuzzy.any_close(terms, f"{r['subject']} {r['sender']}")]
                approximate = bool(rows)

        rows.sort(key=lambda r: r["iso"], reverse=True)
        total = len(rows)
        offset = max(0, int(offset or 0))
        page = rows[offset:offset + _MAX_RESULTS]

        items = []
        for r in page:
            sid = self._handle_for(r["rfc"], r["numeric"], r["account"], {
                "subject": r["subject"], "sender": r["sender"],
                "mailbox": r["mailbox"], "iso": r["iso"],
            })
            items.append({
                "id": sid,
                "from": r["sender"],
                "subject": r["subject"] or "(no subject)",
                "when": _when(r["iso"]),
                "date": r["iso"][:10],
                "read": not r["unread"],
                "folder": r["mailbox"],
                "account": r["account"],
            })

        # A mailbox that emitted exactly the per-mailbox cap was probably
        # truncated, so the total is a lower bound — say "N+" rather than "N".
        per_box: dict[tuple, int] = {}
        for r in rows:
            k = (r["mailbox"], r["account"])
            per_box[k] = per_box.get(k, 0) + 1
        plus = "+" if any(v >= _PER_MAILBOX_CAP for v in per_box.values()) else ""
        result = {"total_matched": f"{total}{plus}" if plus else total, "emails": items}
        if items:
            result["showing"] = (f"{offset + 1}-{offset + len(items)} of {total}{plus} "
                                 "matching emails")
        if approximate:
            result["approximate"] = True
            result["note"] = (f"no exact matches for {query!r} — these are close "
                              "matches (likely a spelling variant); confirm with the "
                              "user before acting on one")
        if inbox_only:
            result["scope"] = "Inbox"
        elif folder:
            result["scope"] = f"folders matching '{folder}'"
        if not items:
            if inbox_only:
                result["note"] = ("the Inbox has nothing matching — add keywords to "
                                  "search every mailbox (Archive included), or widen "
                                  "the days window")
            elif folder:
                result["note"] = (f"nothing found in a folder matching '{folder}' — "
                                  "the folder may be named differently; common names "
                                  "are Archive, Sent, Trash, Junk, Drafts")
            else:
                result["note"] = ("no emails matched — keywords are matched literally, "
                                  "so check the spelling (a dictated name is often "
                                  "spelled differently, e.g. Quantas vs Qantas) or use a "
                                  "shorter distinctive stem of the word; also consider "
                                  "fewer keywords or a date range")
        elif offset + len(page) < total:
            result["more"] = (f"{total - offset - len(page)} older match(es) remain — "
                              f"search again with offset={offset + len(page)}")
        return result

    # ---- read ------------------------------------------------------------

    async def read(self, sid: str, offset: int = 0) -> dict:
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        offset = max(0, int(offset or 0))
        logger.info(f"read_email id={sid} offset={offset}")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
set US to "{_US}"
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set toList to ""
	try
		repeat with r in (to recipients of m)
			set toList to toList & (address of r) & ", "
		end repeat
	end try
	set att to ""
	try
		repeat with x in (mail attachments of m)
			set att to att & (name of x) & ", "
		end repeat
	end try
	return (my clean(subject of m)) & US & (my clean(sender of m)) & US & (my clean(toList)) & US & (my isoDate(date received of m)) & US & (my clean(att)) & US & (content of m)
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not read email: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email is no longer where it was found (moved or "
                             "deleted) — search again"}
        parts = raw.split(_US, 5)
        if len(parts) < 6:
            return {"error": "could not parse the email"}
        subject, sender, to, iso, att, body = parts
        body = body.rstrip("\n")
        total = len(body)
        if offset >= total and total > 0:
            return {"error": f"offset {offset} is past the end ({total} chars)"}
        chunk = body[offset:offset + _PAGE]
        end = offset + len(chunk)
        result = {
            "subject": subject.strip() or "(no subject)",
            "from": sender.strip(),
            "date": iso[:10],
            "folder": entry.get("mailbox", ""),
            "account": entry.get("account", ""),
            "content": chunk,
        }
        if to.strip():
            result["to"] = to.strip().rstrip(",")
        if att.strip():
            result["attachments"] = att.strip().rstrip(",")
        if end < total:
            result["continue_offset"] = end
            result["note"] = (f"{total - end} more characters — call read_email again "
                              f"with the same id and offset={end}")
        return result

    # ---- drafts (compose on screen, send by draft id) ---------------------
    #
    # Drafting and sending are separate tools so nothing goes out unseen: the
    # draft_* calls open a REAL Mail compose window on screen (the user reviews
    # or hand-edits it there), hand back a "d1"-style id, and send_email fires
    # that window off later. The compose window lives in Mail itself, so the id
    # stays valid across tool calls; we re-find it by Mail's outgoing-message id.

    @staticmethod
    def _recipients_block(kind: str, addrs: str) -> str:
        lines = []
        for a in str(addrs or "").replace(";", ",").split(","):
            a = a.strip()
            if a:
                lines.append(f'\t\tmake new {kind} recipient at end of {kind} recipients '
                             f'with properties {{address:"{_osa(a)}"}}')
        return "\n".join(lines)

    @staticmethod
    def _find_out_handler(out_id: int) -> str:
        # Enumerating `outgoing messages` can fail transiently right after
        # another script touched a compose window — retry briefly before
        # concluding anything.
        return f'''
on findOut()
	tell application "Mail"
		repeat 4 times
			try
				repeat with om in outgoing messages
					try
						if (id of om) is {int(out_id)} then return om
					end try
				end repeat
				return missing value
			on error
				delay 0.4
			end try
		end repeat
		return missing value
	end tell
end findOut
'''

    def _draft_entry(self, draft_id: str) -> dict | None:
        return self._drafts.get((draft_id or "").strip())

    @staticmethod
    def _gone(draft_id: str) -> dict:
        return {"error": f"draft {draft_id} is no longer open in Mail (closed or "
                         "already sent) — create it again"}

    def _new_draft_id(self, info: dict) -> str:
        self._draft_counter += 1
        did = f"d{self._draft_counter}"
        self._drafts[did] = info
        while len(self._drafts) > 20:
            self._drafts.popitem(last=False)
        return did

    async def draft_new(self, to: str = "", subject: str = "", body: str = "",
                        cc: str = "", bcc: str = "", from_id: str = "",
                        draft_id: str = "") -> dict:
        if draft_id:
            return await self._edit_draft(draft_id, to, subject, body, cc, bcc)
        if not str(to or "").strip():
            return {"error": "need at least one recipient in 'to'"}
        logger.info(f"draft_new_email to={to!r} subject={subject!r} from_id={from_id!r}")
        sender_line = ""
        from_account = ""
        if str(from_id or "").strip():
            entry = self._resolve(from_id)
            if not entry:
                return self._unknown_handle(from_id)
            from_account = entry.get("account", "")
            if from_account:
                sender_line = (f'\ttry\n\t\tset sender of msg to item 1 of '
                               f'(email addresses of account "{_osa(from_account)}")\n\tend try')
        recips = "\n".join(filter(None, [
            self._recipients_block("to", to),
            self._recipients_block("cc", cc),
            self._recipients_block("bcc", bcc),
        ]))
        script = f'''tell application "Mail"
	set msg to make new outgoing message with properties {{subject:"{_osa(subject)}", content:"{_osa(body)}", visible:true}}
	tell msg
{recips}
	end tell
{sender_line}
	activate
	delay 0.3
	set wid to "0"
	try
		set wid to (id of window 1) as text
	end try
	return ((id of msg) as text) & "{_US}" & wid
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not create the draft: {exc}"}
        out_s, _, win_s = raw.strip().partition(_US)
        try:
            out_id = int(out_s)
        except ValueError:
            return {"error": "Mail did not return a draft id"}
        did = self._new_draft_id({"kind": "new", "out_id": out_id,
                                  "win_id": int(win_s or 0),
                                  "to": to, "subject": subject,
                                  "from_account": from_account})
        result = {"draft_id": did, "to": to, "subject": subject,
                  "open_on_screen": True,
                  "note": ("the draft is open in Mail for review — after the user "
                           f"confirms, call send_email with draft_id '{did}'")}
        if from_account:
            result["from_account"] = from_account
        return result

    async def draft_reply(self, sid: str, body: str, reply_all: bool = False,
                          draft_id: str = "") -> dict:
        if draft_id:
            return await self._edit_draft(draft_id, "", "", body, "", "")
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        logger.info(f"draft_reply_email id={sid} reply_all={reply_all}")
        verb = ("reply m with opening window and reply to all" if reply_all
                else "reply m with opening window")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set r to {verb}
	set content of r to "{_osa(body)}"
	activate
	delay 0.3
	set wid to "0"
	try
		set wid to (id of window 1) as text
	end try
	return ((id of r) as text) & "{_US}" & wid
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not draft the reply: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email is no longer available to reply to — search again"}
        out_s, _, win_s = raw.strip().partition(_US)
        try:
            out_id = int(out_s)
        except ValueError:
            return {"error": "Mail did not return a draft id"}
        did = self._new_draft_id({"kind": "reply", "out_id": out_id,
                                  "win_id": int(win_s or 0),
                                  "to": entry.get("sender", ""),
                                  "subject": entry.get("subject", "")})
        return {"draft_id": did, "replying_to": entry.get("subject", ""),
                "reply_all": bool(reply_all), "open_on_screen": True,
                "note": ("the reply draft is open in Mail for review — after the "
                         f"user confirms, call send_email with draft_id '{did}'")}

    async def _edit_draft(self, draft_id: str, to: str, subject: str,
                          body: str, cc: str, bcc: str) -> dict:
        info = self._draft_entry(draft_id)
        if not info:
            return {"error": f"unknown draft id '{draft_id}' — draft_new_email or "
                             "draft_reply_email returns one"}
        logger.info(f"edit draft {draft_id}")
        sets = []
        if str(subject or "").strip() and info["kind"] == "new":
            sets.append(f'\tset subject of om to "{_osa(subject)}"')
        if str(body or "").strip():
            sets.append(f'\tset content of om to "{_osa(body)}"')
        for kind, addrs in (("to", to), ("cc", cc), ("bcc", bcc)):
            if str(addrs or "").strip() and info["kind"] == "new":
                sets.append(f"\tdelete every {kind} recipient of om")
                block = self._recipients_block(kind, addrs).replace("\t\t", "\t\t")
                sets.append(f"\ttell om\n{block}\n\tend tell")
        if not sets:
            return {"error": "nothing to change — pass the field(s) to update"}
        script = f'''{self._find_out_handler(info["out_id"])}
tell application "Mail"
	set om to my findOut()
	if om is missing value then return "GONE"
{chr(10).join(sets)}
	activate
	return "OK"
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not edit the draft: {exc}"}
        if raw.strip() == "GONE":
            self._drafts.pop(draft_id, None)
            return self._gone(draft_id)
        if str(subject or "").strip() and info["kind"] == "new":
            info["subject"] = subject
        if str(to or "").strip() and info["kind"] == "new":
            info["to"] = to
        return {"draft_id": draft_id, "updated": True, "open_on_screen": True,
                "note": ("draft updated on screen — after the user confirms, call "
                         f"send_email with draft_id '{draft_id}'")}

    async def discard_draft(self, draft_id: str) -> dict:
        """Discard a draft: close its on-screen compose window without saving.

        Outgoing messages can't be `delete`d via AppleScript — closing their
        compose window (saving no) is the supported way to discard one, which is
        why the window id is captured at draft creation.
        """
        info = self._draft_entry(draft_id)
        if not info:
            return {"error": f"unknown draft id '{draft_id}' — nothing to discard"}
        logger.info(f"discard_draft {draft_id}")
        win_id = int(info.get("win_id") or 0)
        if not win_id:
            return {"error": "this draft's window could not be identified — close "
                             "the compose window in Mail by hand; nothing was sent"}
        # Mail's compose windows ignore `saving no`/`saving yes` and throw a
        # "save this draft?" sheet regardless (Save / Don't Save / Cancel). So:
        # close, then click the sheet's "Don't Save" button via Accessibility —
        # the same permission the notification watcher already requires.
        subj = info.get("subject", "")
        script = f'''tell application "Mail"
	try
		close (window id {win_id}) saving no
	on error
		return "GONE"
	end try
end tell
delay 0.5
tell application "System Events" to tell process "Mail"
	repeat 4 times
		repeat with w in windows
			try
				if exists sheet 1 of w then
					repeat with b in buttons of sheet 1 of w
						set bn to name of b
						if bn starts with "Don" or bn starts with "Nicht" then
							click b
							delay 0.3
							return "DISCARDED"
						end if
					end repeat
				end if
			end try
		end repeat
		delay 0.4
	end repeat
	return "NOSHEET"
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not discard the draft: {exc}"}
        self._drafts.pop(draft_id, None)
        status = raw.strip()
        if status == "GONE":
            return {"discarded": True, "note": "the draft window was already closed"}
        if status == "NOSHEET":
            # Window closed without asking (empty/unchanged draft) — also fine.
            return {"discarded": True, "subject": subj,
                    "note": "draft closed — nothing was sent"}
        return {"discarded": True, "subject": subj,
                "note": "draft discarded and its window closed without saving — "
                        "nothing was sent"}

    async def send_draft(self, draft_id: str) -> dict:
        info = self._draft_entry(draft_id)
        if not info:
            return {"error": f"unknown draft id '{draft_id}' — draft_new_email or "
                             "draft_reply_email returns one; nothing was sent"}
        logger.info(f"send_email draft={draft_id}")
        script = f'''{self._find_out_handler(info["out_id"])}
tell application "Mail"
	set om to my findOut()
	if om is missing value then return "GONE"
	send om
	return "SENT"
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not send: {exc}"}
        if raw.strip() == "GONE":
            self._drafts.pop(draft_id, None)
            return self._gone(draft_id)
        self._drafts.pop(draft_id, None)
        return {"sent": True, "to": info.get("to", ""),
                "subject": info.get("subject", ""),
                **({"was_reply": True} if info["kind"] == "reply" else {})}

    # ---- archive ---------------------------------------------------------

    async def archive(self, sid: str) -> dict:
        """Mark read and move to the message's account's Archive mailbox."""
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        logger.info(f"archive_email id={sid}")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set subj to my clean(subject of m)
	try
		set read status of m to true
	end try
	set acct to account of mailbox of m
	set dest to missing value
	repeat with destName in {{"Archive", "All Mail", "Archiv"}}
		try
			set dest to mailbox (destName as text) of acct
			exit repeat
		end try
	end repeat
	if dest is missing value then return "NOARCHIVE"
	move m to dest
	return subj
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not archive: {exc}"}
        raw = raw.strip()
        if raw == "NOTFOUND":
            return {"error": "that email is already gone (moved or deleted)"}
        if raw == "NOARCHIVE":
            return {"error": "this account has no Archive mailbox — mark_email it "
                             "read instead, or trash_email it"}
        return {"archived": True, "marked_read": True,
                "subject": raw or entry.get("subject", "")}

    # ---- trash -----------------------------------------------------------

    async def trash(self, sid: str) -> dict:
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        logger.info(f"trash_email id={sid}")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set subj to my clean(subject of m)
	delete m
	return subj
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not trash: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email is already gone (moved or deleted)"}
        return {"trashed": True, "subject": raw.strip() or entry.get("subject", "")}

    # ---- mark ------------------------------------------------------------

    async def mark(self, sid: str, status: str) -> dict:
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        status = (status or "").strip().lower()
        setter = {
            "read": "set read status of m to true",
            "unread": "set read status of m to false",
            "flagged": "set flagged status of m to true",
            "unflagged": "set flagged status of m to false",
        }.get(status)
        if not setter:
            return {"error": "status must be one of: read, unread, flagged, unflagged"}
        logger.info(f"mark_email id={sid} status={status}")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	{setter}
	return "OK"
end tell'''
        try:
            raw = await self._run(script)
        except RuntimeError as exc:
            return {"error": f"could not update: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email is no longer available — search again"}
        return {"marked": status, "subject": entry.get("subject", "")}
