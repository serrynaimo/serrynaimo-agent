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
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from loguru import logger

import fuzzy

# Field / record separators — ASCII unit + record separator, never in mail text.
_US, _RS, _GS = "\x1f", "\x1e", "\x1d"

# Per-call osascript timeout (seconds). Reading one body can be slow on IMAP.
_TIMEOUT = 90
# Max characters of body per read() page — keeps a single tool result small.
_PAGE = 4000
# Newest-first result cap for search.
_MAX_RESULTS = 40
# Safety cap on how many matches we pull per mailbox before sorting.
_PER_MAILBOX_CAP = 150
# Consecutive keyword searches within this many seconds reuse the previous
# column dump outright (read flags may lag that long; ids/subjects/senders
# only ever gain new rows, which a fresh dump next search picks up).
_SCAN_TTL = 45
# How many resolved handles to remember.
_CACHE_MAX = 400

# System folders we never search (noise + slow); default case-insensitive match.
_SKIP_MAILBOXES = {
    "trash", "deleted messages", "deleted items", "bin",
    "junk", "junk e-mail", "spam",
    "sent", "sent messages", "sent mail", "sent items",
    "drafts", "outbox",
}


def _fold(s) -> str:
    """Fold curly quotes to ASCII so keyword matching ignores smart punctuation."""
    s = str(s)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return s


def _osa(s) -> str:
    """Escape a Python value for an AppleScript double-quoted literal.

    Folds curly quotes to ASCII (so keywords match smart-punctuation subjects),
    then escapes backslash/quote and turns newlines/tabs into AppleScript's own
    \\n / \\t escapes (which it does interpret inside string literals).
    """
    s = _fold(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n").replace("\t", "\\t")
    return s


def _scrub(s: str) -> str:
    """Clean a column-dump field for display: drop AppleScript's 'missing
    value' placeholder and collapse whitespace/control characters."""
    if s == "missing value":
        return ""
    return " ".join(s.replace(_US, " ").replace(_RS, " ").replace(_GS, " ").split())


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
        # Fast-scan caches, keyed by (account, mailbox). _scan_cols holds the
        # id/subject/sender columns of the last dump (subjects and senders are
        # immutable per message id, so they are reused whenever a fresh id
        # dump matches). _scan_detail maps numeric id -> (rfc, iso), both
        # immutable, filled lazily for matched messages. _scan_snapshot
        # short-circuits the whole dump for _SCAN_TTL seconds.
        self._scan_cols: dict[tuple[str, str], dict] = {}
        self._scan_detail: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
        self._scan_snapshot: tuple[float, list] | None = None

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

    async def warmup(self) -> None:
        """Pre-fill the search column caches (e.g. at bot start) so the first
        keyword search skips the cold whole-mailbox dump."""
        await self._dump_columns("")

    async def _dump_columns(self, folder: str) -> list[tuple]:
        """Dump per-mailbox (id, read) columns, with subject/sender columns
        served from _scan_cols when the id column is unchanged.

        Mailboxes not yet in the cache get their subject/sender columns in
        the same script; known-but-changed ones are re-fetched in a second,
        smaller script. Returns (account, mailbox, ids, subjects, senders,
        reads) tuples; a mailbox whose dump raced a new arrival (column
        counts disagree) sits out this pass and self-heals on the next.
        """
        if folder:
            fl = _osa(folder)
            select = f'set doScan to (mbName is "{fl}" or mbName starts with "{fl}")'
        else:
            select = '''if isGmail then
					set doScan to (mbName is "INBOX" or mbName is "All Mail")
				else
					set doScan to (skipNames does not contain mbName)
				end if'''
        skip = "{" + ", ".join(f'"{_osa(n)}"' for n in sorted(_SKIP_MAILBOXES)) + "}"
        known = "{" + ", ".join(
            f'"{_osa(acc)}{_US}{_osa(mb)}"' for acc, mb in self._scan_cols) + "}"
        script = f'''
set US to "{_US}"
set RS to "{_RS}"
set GS to "{_GS}"
set skipNames to {skip}
set knownKeys to {known}
tell application "Mail"
	set outL to {{}}
	repeat with a in every account
		set accName to name of a
		set srv to ""
		try
			set srv to server name of a
		end try
		set isGmail to (srv contains "gmail") or (srv contains "google")
		repeat with mb in (every mailbox of a)
			set mbName to ""
			try
				set mbName to name of mb
			end try
			set doScan to false
			try
				{select}
			end try
			if doScan then
				try
					set idL to id of every message of mb
					set readL to read status of every message of mb
					set n0 to count of idL
					if n0 is (count of readL) then
						set AppleScript's text item delimiters to US
						set hdr to "" & accName & US & mbName & US & (n0 as text)
						if knownKeys contains (accName & US & mbName) then
							set end of outL to hdr & RS & (idL as string) & RS & (readL as string)
						else
							set subjL to subject of every message of mb
							set sndL to sender of every message of mb
							if (n0 is (count of subjL)) and (n0 is (count of sndL)) then
								set end of outL to hdr & RS & (idL as string) & RS & (readL as string) & RS & (subjL as string) & RS & (sndL as string)
							end if
						end if
					end if
				end try
			end if
		end repeat
	end repeat
	set AppleScript's text item delimiters to GS
	return outL as string
end tell'''
        raw = (await self._run(script)).strip("\r\n")

        # (account, mailbox, ids, reads) per scanned mailbox
        dumped: list[tuple[str, str, list, list]] = []
        for sec in raw.split(_GS):
            parts = sec.split(_RS)
            if len(parts) not in (3, 5):
                continue
            h = parts[0].split(_US)
            if len(h) != 3 or not h[2].isdigit():
                continue
            acc, mbname, n = h[0], h[1], int(h[2])
            if n <= 0:
                continue
            cols = [c.split(_US) for c in parts[1:]]
            if any(len(c) != n for c in cols):
                # A field contained a delimiter char, or mail arrived mid-dump.
                logger.warning(f"search_email column mismatch in {acc}/{mbname} "
                               "— mailbox skipped this pass")
                continue
            if len(parts) == 5:
                self._scan_cols[(acc, mbname)] = {
                    "ids": cols[0], "subj": cols[2], "snd": cols[3]}
            dumped.append((acc, mbname, cols[0], cols[1]))

        # Known mailboxes whose id column changed since the cached dump need
        # fresh subject/sender columns — one more script, changed boxes only.
        need = [(acc, mbname) for acc, mbname, ids, _ in dumped
                if self._scan_cols.get((acc, mbname), {}).get("ids") != ids]
        if need:
            segs = []
            for acc, mbname in need:
                segs.append(f'''
	set mb to missing value
	try
		set a to account "{_osa(acc)}"
		repeat with mbx in (every mailbox of a)
			try
				if (name of mbx) is "{_osa(mbname)}" then
					set mb to mbx
					exit repeat
				end if
			end try
		end repeat
	end try
	if mb is not missing value then
		try
			set idL to id of every message of mb
			set subjL to subject of every message of mb
			set sndL to sender of every message of mb
			if ((count of idL) is (count of subjL)) and ((count of idL) is (count of sndL)) then
				set AppleScript's text item delimiters to US
				set hdr to "" & "{_osa(acc)}" & US & "{_osa(mbname)}" & US & ((count of idL) as text)
				set end of outL to hdr & RS & (idL as string) & RS & (subjL as string) & RS & (sndL as string)
			end if
		end try
	end if''')
            script_b = f'''
set US to "{_US}"
set RS to "{_RS}"
set GS to "{_GS}"
tell application "Mail"
	set outL to {{}}
{"".join(segs)}
	set AppleScript's text item delimiters to GS
	return outL as string
end tell'''
            raw_b = (await self._run(script_b)).strip("\r\n")
            for sec in raw_b.split(_GS):
                parts = sec.split(_RS)
                if len(parts) != 4:
                    continue
                h = parts[0].split(_US)
                if len(h) != 3 or not h[2].isdigit():
                    continue
                acc, mbname, n = h[0], h[1], int(h[2])
                cols = [c.split(_US) for c in parts[1:]]
                if any(len(c) != n for c in cols):
                    continue
                self._scan_cols[(acc, mbname)] = {
                    "ids": cols[0], "subj": cols[1], "snd": cols[2]}

        boxes: list[tuple] = []
        for acc, mbname, ids, reads in dumped:
            cached = self._scan_cols.get((acc, mbname))
            if not cached or cached["ids"] != ids:
                continue
            boxes.append((acc, mbname, ids, cached["subj"], cached["snd"],
                          reads))
        if not folder:
            self._scan_snapshot = (time.monotonic(), boxes)
        return boxes

    async def _scan_fast(self, terms: list[str], unread_only: bool,
                         folder: str) -> tuple[list[dict], bool]:
        """Keyword scan via whole-column dumps filtered in Python.

        Mail's `whose` predicate is the slow part of searching (3-5s on a
        20k-message mailbox for a handful of hits), while a plain
        `subject of every message` dump is ~1s — and range specifiers
        (`messages 1 thru N`) are even slower than `whose`, so dumps are
        always whole-mailbox. The flow:

        1. Reuse the previous dump wholesale if it is under _SCAN_TTL old.
        2. Otherwise dump id + read-status columns per mailbox — plus
           subject/sender for mailboxes never seen before — and, where a
           known mailbox's id column changed, re-fetch its subject/sender
           columns in a second pass.
        3. Match keywords in Python (the fuzzy retry is free — same columns).
        4. Fetch date + RFC id for matches only — by position, verified
           against the numeric id (new arrivals shift positions) — with an
           (account, mailbox, id) cache in front, since both are immutable.

        Gmail accounts scan only INBOX + All Mail — every other Gmail
        "mailbox" is a label, i.e. a subset of All Mail.

        Returns (rows, approximate) where rows match scan()'s shape.
        """
        boxes: list[tuple[str, str, list, list, list, list]] | None = None
        if not folder and self._scan_snapshot:
            ts, snap = self._scan_snapshot
            if time.monotonic() - ts < _SCAN_TTL:
                boxes = snap
        if boxes is None:
            boxes = await self._dump_columns(folder)

        def match(words: list[str], verify_close: bool) -> list[tuple[int, int]]:
            norm = [_fold(w).lower() for w in words]
            picked = []
            for bi, (acc, mbname, ids, subjs, snds, reads) in enumerate(boxes):
                per_box = 0
                for i in range(len(ids)):
                    if unread_only and reads[i] != "false":
                        continue
                    blob = f"{subjs[i]} {snds[i]}"
                    hay = _fold(blob).lower()
                    if not any(w in hay for w in norm):
                        continue
                    if verify_close and not fuzzy.any_close(terms, blob):
                        continue
                    picked.append((bi, i))
                    per_box += 1
                    if per_box >= _PER_MAILBOX_CAP:  # position 1 = newest
                        break
            return picked

        picked = match(terms, False)
        approximate = False
        if not picked:
            stems: list[str] = []
            for t in terms:
                for v in fuzzy.variants(t):
                    if v not in stems:
                        stems.append(v)
            if stems:
                logger.info(f"search_email fuzzy retry with stems {stems}")
                picked = match(stems, True)
                approximate = bool(picked)
        if not picked:
            return [], False

        # Phase 2: date received + RFC id for matches only — both immutable,
        # so cached hits skip Mail entirely. Uncached ones are addressed by
        # position and verified against the numeric id (positions shift when
        # mail arrives between the dump and this fetch — mismatches are
        # dropped rather than misattributed).
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for bi, i in picked:
            acc, mbname, ids = boxes[bi][0], boxes[bi][1], boxes[bi][2]
            if not ids[i].isdigit():
                continue
            if ids[i] in self._scan_detail.get((acc, mbname), {}):
                continue
            groups.setdefault((acc, mbname), []).append((i + 1, int(ids[i])))
        segs = []
        for (acc, mbname), pairs in groups.items():
            pl = ", ".join(f"{{{p}, {mid}}}" for p, mid in pairs)
            segs.append(f'''
	set mb to missing value
	try
		set a to account "{_osa(acc)}"
		repeat with mbx in (every mailbox of a)
			try
				if (name of mbx) is "{_osa(mbname)}" then
					set mb to mbx
					exit repeat
				end if
			end try
		end repeat
	end try
	if mb is not missing value then
		set refs to every message of mb
		set nRefs to count of refs
		repeat with pr in {{{pl}}}
			try
				set p to item 1 of pr
				if p is not greater than nRefs then
					set m to item p of refs
					if (id of m) is (item 2 of pr) then
					set rfcId to ""
					try
						set rfcId to (message id of m) as text
					end try
						set end of outL to ((item 2 of pr) as text) & US & (my clean(rfcId)) & US & (my isoDate(date received of m)) & US & "{_osa(acc)}" & US & "{_osa(mbname)}"
					end if
				end if
			end try
		end repeat
	end if''')
        if segs:
            script2 = f'''{_PRELUDE}
set US to "{_US}"
set RS to "{_RS}"
tell application "Mail"
	set outL to {{}}
{"".join(segs)}
	set AppleScript's text item delimiters to RS
	return outL as string
end tell'''
            raw2 = (await self._run(script2)).strip("\r\n")
            for rec in raw2.split(_RS):
                f = rec.split(_US)
                if len(f) != 5:
                    continue
                nid, rfc, iso, acc, mbname = f
                self._scan_detail.setdefault((acc, mbname), {})[nid] = (rfc, iso)

        rows: list[dict] = []
        seen: dict[str, int] = {}
        for bi, i in picked:
            acc, mbname, ids, subjs, snds, reads = boxes[bi]
            d = self._scan_detail.get((acc, mbname), {}).get(ids[i])
            if not d:
                continue
            rfc, iso = d
            row = {
                "numeric": ids[i], "rfc": rfc, "subject": _scrub(subjs[i]),
                "sender": _scrub(snds[i]), "mailbox": mbname, "account": acc,
                "unread": reads[i] == "false", "iso": iso,
            }
            key = rfc.strip() or f"{ids[i]}:{acc}"
            prev = seen.get(key)
            if prev is None:
                seen[key] = len(rows)
                rows.append(row)
            elif (mbname.upper() == "INBOX"
                  and rows[prev]["mailbox"].upper() != "INBOX"):
                rows[prev] = row  # prefer the Inbox-labelled copy (Gmail dupes)
        return rows, approximate

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
                # Gmail mailboxes other than INBOX are labels — subsets of All
                # Mail — so scanning them only re-finds the same messages.
                mb_cond = ('((isGmail and (mbName is "INBOX" or mbName is '
                           '"All Mail")) or ((not isGmail) and '
                           '(skipNames does not contain mbName)))')
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
		set srv to ""
		try
			set srv to server name of a
		end try
		set isGmail to (srv contains "gmail") or (srv contains "google")
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
            seen: dict[str, int] = {}
            for rec in raw.split(_RS):
                if not rec.strip():
                    continue
                f = rec.split(_US)
                if len(f) < 8:
                    continue
                numeric, rfc, subject, sender, mailbox, account, read, iso = f[:8]
                row = {
                    "numeric": numeric, "rfc": rfc, "subject": subject.strip(),
                    "sender": sender.strip(), "mailbox": mailbox, "account": account,
                    "unread": read == "0", "iso": iso,  # AppleScript emits "1" = read
                }
                dedup = rfc.strip() or f"{numeric}:{account}"
                prev = seen.get(dedup)
                if prev is None:
                    seen[dedup] = len(rows)
                    rows.append(row)
                elif (mailbox.strip().upper() == "INBOX"
                      and rows[prev]["mailbox"].strip().upper() != "INBOX"):
                    rows[prev] = row  # prefer the Inbox-labelled copy (Gmail dupes)
            return rows

        # Keyword searches without date conditions take the fast column-dump
        # path (see _scan_fast); date windows still need `whose` so the
        # per-mailbox cap applies after date filtering.
        approximate = False
        if terms and not (days and days > 0) and not date_conds:
            try:
                rows, approximate = await self._scan_fast(terms, unread_only,
                                                          folder)
            except RuntimeError as exc:
                return {"error": f"could not search mail: {exc}"}
        else:
            try:
                rows = await scan(terms)
            except RuntimeError as exc:
                return {"error": f"could not search mail: {exc}"}

            # Fuzzy fallback: zero literal hits often just means a misspelled
            # name (dictation writes "Quantas" for Qantas). Retry once with
            # substring stems of the terms, then keep only candidates whose
            # sender/subject actually resembles what was asked — labelled
            # approximate below.
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
                            if fuzzy.any_close(terms,
                                               f"{r['subject']} {r['sender']}")]
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
set RS to "{_RS}"
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
			set att to att & (my clean(name of x)) & RS
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
        att_names = [n.strip() for n in att.split(_RS) if n.strip()]
        if att_names:
            result["attachments"] = att_names
        if end < total:
            result["continue_offset"] = end
            result["note"] = (f"{total - end} more characters — call read_email again "
                              f"with the same id and offset={end}")
        return result

    # ---- attachments -------------------------------------------------------

    @staticmethod
    def _unique_target(directory: Path, name: str, taken: set[str]) -> Path:
        """A collision-free path in `directory` for `name` (never overwrites)."""
        name = name.strip().replace("/", "-").replace("\x00", "") or "attachment"
        stem, dot, ext = name.rpartition(".")
        if not dot or not stem:
            stem, ext = name, ""
        n = 1
        candidate = name
        while candidate.lower() in taken or (directory / candidate).exists():
            n += 1
            candidate = f"{stem}-{n}.{ext}" if ext else f"{name}-{n}"
        taken.add(candidate.lower())
        return directory / candidate

    async def save_attachment(self, sid: str, name: str = "") -> dict:
        """Save one attachment (by name) — or all of them — to ~/Downloads.

        Never overwrites: an existing file gets a `-2` style suffix. Two
        osascript passes: list the attachment names first so Python can pick
        collision-free targets, then save each by exact (cleaned) name.
        """
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        name = str(name or "").strip()
        logger.info(f"save_attachment id={sid} name={name!r}")
        list_script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
set RS to "{_RS}"
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set out to ""
	repeat with x in (mail attachments of m)
		try
			set out to out & (my clean(name of x)) & RS
		end try
	end repeat
	return out
end tell'''
        try:
            raw = await self._run(list_script)
        except RuntimeError as exc:
            return {"error": f"could not read the email's attachments: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email is no longer where it was found (moved or "
                             "deleted) — search again"}
        names = [n.strip() for n in raw.split(_RS) if n.strip()]
        if not names:
            return {"error": "that email has no attachments"}
        if name:
            wanted = [n for n in names if n.lower() == name.lower()]
            if not wanted:
                wanted = [n for n in names if name.lower() in n.lower()]
            if not wanted:
                return {"error": f"no attachment named {name!r} — this email has: "
                                 + ", ".join(names)}
            wanted = wanted[:1]
        else:
            wanted = list(dict.fromkeys(names))  # all of them, deduped by name

        downloads = Path.home() / "Downloads"
        taken: set[str] = set()
        targets = {n: self._unique_target(downloads, n, taken) for n in wanted}
        saves = "\n".join(
            f'''		if (not isDone) and nm is "{_osa(n)}" then
			set isDone to true
			try
				save x in (POSIX file "{_osa(str(p))}")
				set out to out & "OK" & US & nm & RS
			on error errMsg
				set out to out & "ERR" & US & nm & US & (my clean(errMsg)) & RS
			end try
		end if'''
            for n, p in targets.items())
        save_script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
set US to "{_US}"
set RS to "{_RS}"
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set out to ""
	repeat with x in (mail attachments of m)
		set nm to my clean(name of x)
		set isDone to false
{saves}
	end repeat
	return out
end tell'''
        try:
            raw = await self._run(save_script)
        except RuntimeError as exc:
            return {"error": f"could not save: {exc}"}
        if raw.strip() == "NOTFOUND":
            return {"error": "that email disappeared mid-save (moved or deleted) — "
                             "search again"}
        saved, failed = [], []
        for rec in raw.split(_RS):
            f = rec.split(_US)
            if len(f) >= 2 and f[0] == "OK":
                saved.append(str(targets[f[1]]))
            elif len(f) >= 2 and f[0] == "ERR":
                failed.append({"name": f[1], "reason": f[2] if len(f) > 2 else "unknown"})
        result: dict = {}
        if saved:
            result["saved"] = saved
            result["folder"] = str(downloads)
        if failed:
            result["failed"] = failed
            result["note"] = ("a failed save usually means Mail hasn't downloaded "
                              "the attachment yet — open the email in Mail once, "
                              "then try again")
        if not saved and not failed:
            return {"error": "nothing was saved — the attachments changed under us; "
                             "read_email again to see the current list"}
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

    @staticmethod
    def _attach_lines(var: str, path: str) -> str:
        """AppleScript lines appending a file attachment to outgoing message `var`."""
        if not path:
            return ""
        return (f'\tdelay 0.5\n'
                f'\ttell content of {var}\n'
                f'\t\tmake new attachment with properties '
                f'{{file name:(POSIX file "{_osa(path)}")}} at after the last paragraph\n'
                f'\tend tell')

    @staticmethod
    def _attachment_file(attachment_path: str) -> tuple[str, dict | None]:
        """Expand and validate an attachment path; ('', error-dict) if bad."""
        raw = str(attachment_path or "").strip()
        if not raw:
            return "", None
        p = Path(raw).expanduser()
        if not p.is_file():
            return "", {"error": f"attachment file not found: {p} — pass the full "
                                 "path of an existing file (~/ is fine)"}
        return str(p.resolve()), None

    async def draft_email(self, to: str = "", subject: str = "", body: str = "",
                          cc: str = "", bcc: str = "", from_id: str = "",
                          reply_to_email_id: str = "", reply_all: bool = False,
                          attachment_path: str = "", draft_id: str = "") -> dict:
        """One entry point for drafting: new email, reply, or edit-by-draft-id."""
        attach, err = self._attachment_file(attachment_path)
        if err:
            return err
        if str(draft_id or "").strip():
            return await self._edit_draft(draft_id, to, subject, body, cc, bcc, attach)
        if str(reply_to_email_id or "").strip():
            return await self._draft_reply(reply_to_email_id, body, reply_all, attach)
        return await self._draft_new(to, subject, body, cc, bcc, from_id, attach)

    async def _draft_new(self, to: str, subject: str, body: str,
                         cc: str, bcc: str, from_id: str, attach: str) -> dict:
        if not str(to or "").strip():
            return {"error": "need at least one recipient in 'to' (or pass "
                             "reply_to_email_id to reply to an email)"}
        logger.info(f"draft_email to={to!r} subject={subject!r} from_id={from_id!r} "
                    f"attach={attach!r}")
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
{self._attach_lines("msg", attach)}
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
        if attach:
            result["attached"] = Path(attach).name
        if from_account:
            result["from_account"] = from_account
        return result

    async def _draft_reply(self, sid: str, body: str, reply_all: bool,
                           attach: str) -> dict:
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        logger.info(f"draft_email reply_to={sid} reply_all={reply_all} attach={attach!r}")
        verb = ("reply m with opening window and reply to all" if reply_all
                else "reply m with opening window")
        script = f'''{_PRELUDE}{_find_handler(entry["clause"], entry["account"])}
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set r to {verb}
	set content of r to "{_osa(body)}"
{self._attach_lines("r", attach)}
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
        result = {"draft_id": did, "replying_to": entry.get("subject", ""),
                  "reply_all": bool(reply_all), "open_on_screen": True,
                  "note": ("the reply draft is open in Mail for review — after the "
                           f"user confirms, call send_email with draft_id '{did}'")}
        if attach:
            result["attached"] = Path(attach).name
        return result

    async def _edit_draft(self, draft_id: str, to: str, subject: str,
                          body: str, cc: str, bcc: str, attach: str = "") -> dict:
        info = self._draft_entry(draft_id)
        if not info:
            return {"error": f"unknown draft id '{draft_id}' — draft_email "
                             "returns one"}
        logger.info(f"edit draft {draft_id} attach={attach!r}")
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
        if attach:
            sets.append(self._attach_lines("om", attach))
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
        result = {"draft_id": draft_id, "updated": True, "open_on_screen": True,
                  "note": ("draft updated on screen — after the user confirms, call "
                           f"send_email with draft_id '{draft_id}'")}
        if attach:
            result["attached"] = Path(attach).name
        return result

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
            return {"error": f"unknown draft id '{draft_id}' — draft_email "
                             "returns one; nothing was sent"}
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
        """Mark read and archive.

        Non-Gmail accounts: move to the account's Archive mailbox. Gmail
        accounts: an AppleScript `move` only ADDS a label — the message keeps
        its Inbox label — so instead select the message in a viewer and invoke
        Mail's own Message > Archive command via UI scripting, which strips
        the Inbox label the way Gmail expects. Briefly brings Mail frontmost
        (the menu is disabled otherwise), then restores the previous app.
        """
        entry = self._resolve(sid)
        if not entry:
            return self._unknown_handle(sid)
        logger.info(f"archive_email id={sid}")
        clause = entry["clause"]
        script = f'''{_PRELUDE}{_find_handler(clause, entry["account"])}
set srv to ""
tell application "Mail"
	set m to my findMsg()
	if m is missing value then return "NOTFOUND"
	set subj to my clean(subject of m)
	set targetRfc to ""
	try
		set targetRfc to message id of m
	end try
	try
		set read status of m to true
	end try
	set acct to account of mailbox of m
	try
		set srv to server name of acct
	end try
end tell
if srv does not contain "gmail" and srv does not contain "google" then
	tell application "Mail"
		set dest to missing value
		repeat with destName in {{"Archive", "All Mail", "Archiv"}}
			try
				set dest to mailbox (destName as text) of acct
				exit repeat
			end try
		end repeat
		if dest is missing value then return "NOARCHIVE"
		move m to dest
		return "OK" & subj
	end tell
end if
tell application "Mail"
	set inb to mailbox "INBOX" of acct
	if (count of (messages of inb whose {clause})) is 0 then return "NOTINBOX" & subj
	if (count of message viewers) is 0 then make new message viewer
	set mv to message viewer 1
	set prevBoxes to selected mailboxes of mv
	set selected mailboxes of mv to {{inb}}
end tell
set frontProc to ""
tell application "System Events"
	try
		set frontProc to name of first application process whose frontmost is true
	end try
end tell
tell application "Mail" to activate
delay 0.8
set outcome to "ROWNOTFOUND"
tell application "System Events" to tell process "Mail"
	set theTable to missing value
	set theWin to missing value
	repeat with w in windows
		try
			set t to UI element 1 of UI element 1 of UI element 4 of splitter group 1 of w
			if role of t is "AXTable" then
				set theTable to t
				set theWin to w
				exit repeat
			end if
		end try
	end repeat
	if theTable is missing value then
		set outcome to "NOVIEWER"
	else
		perform action "AXRaise" of theWin
		delay 0.4
		repeat with rw in rows of theTable
			set hit to false
			try
				repeat with t in UI elements of UI element 1 of UI element 1 of rw
					if role of t is "AXStaticText" and value of t is subj then
						set hit to true
						exit repeat
					end if
				end repeat
			end try
			if hit then
				set selected of rw to true
				delay 0.4
				set verified to (targetRfc is "")
				if not verified then
					tell application "Mail"
						try
							set selMsgs to selected messages of message viewer 1
							if selMsgs is not missing value and (count of selMsgs) is 1 then
								if message id of item 1 of selMsgs is targetRfc then set verified to true
							end if
						end try
					end tell
				end if
				if verified then
					set mi to missing value
					repeat with mname in {{"Message", "Nachricht"}}
						try
							set mi to menu item "Archive" of menu 1 of menu bar item (mname as text) of menu bar 1
							exit repeat
						end try
					end repeat
					if mi is not missing value and enabled of mi then
						click mi
						set outcome to "CLICKED"
					else
						set outcome to "MENUDISABLED"
					end if
					exit repeat
				end if
			end if
		end repeat
	end if
end tell
set gone to false
if outcome is "CLICKED" then
	repeat 10 times
		delay 0.6
		tell application "Mail"
			if (count of (messages of inb whose {clause})) is 0 then
				set gone to true
				exit repeat
			end if
		end tell
	end repeat
end if
tell application "Mail"
	try
		set selected mailboxes of message viewer 1 to prevBoxes
	end try
end tell
if frontProc is not "" and frontProc is not "Mail" then
	try
		tell application "System Events" to set frontmost of process frontProc to true
	end try
end if
if outcome is "CLICKED" then
	if gone then return "OK" & subj
	return "SLOWSYNC" & subj
end if
return outcome'''
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
        if raw.startswith("NOTINBOX"):
            return {"archived": True, "marked_read": True,
                    "subject": raw[len("NOTINBOX"):] or entry.get("subject", ""),
                    "note": "it was already out of the inbox"}
        if raw == "NOVIEWER":
            return {"error": "could not drive the Mail window to archive this "
                             "Gmail message — is a Mail viewer window open?"}
        if raw == "ROWNOTFOUND":
            return {"error": "could not locate this message in the Mail inbox "
                             "list to archive it (Gmail account)"}
        if raw == "MENUDISABLED":
            return {"error": "Mail's Archive command was unavailable for this "
                             "message"}
        if raw.startswith("SLOWSYNC"):
            return {"archived": True, "marked_read": True,
                    "subject": raw[len("SLOWSYNC"):] or entry.get("subject", ""),
                    "note": "archive issued; Gmail may take a moment to sync"}
        if raw.startswith("OK"):
            return {"archived": True, "marked_read": True,
                    "subject": raw[len("OK"):] or entry.get("subject", "")}
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
