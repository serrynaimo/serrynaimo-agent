"""One-time macOS permission bootstrap for the voice agent's MCP toolsets.

Run this from the SAME terminal app you use to run bot.py:

    uv run request_permissions.py

It triggers the two system permission prompts (click Allow on each):
- Calendars (full access) — for the calendar toolset (mcp-ical)
- Mail automation — for the apple-mail toolset
"""

import os
import subprocess

status_names = {0: "notDetermined", 1: "restricted", 2: "denied",
                3: "authorized(legacy)", 4: "fullAccess"}


CALENDAR_MCP = os.path.expanduser("~/Sites/apple-calendar-mcp")
GET_CALENDARS_BIN = os.path.join(
    CALENDAR_MCP, "src", "apple_calendar_mcp", "swift", "bin", "get_calendars"
)
GET_CALENDARS_SWIFT = os.path.join(
    CALENDAR_MCP, "src", "apple_calendar_mcp", "swift", "get_calendars.swift"
)


def calendar():
    """Trigger calendar access exactly the way apple-calendar-mcp does at
    runtime: by running its get_calendars Swift helper."""
    try:
        from EventKit import EKEventStore

        before = EKEventStore.authorizationStatusForEntityType_(0)
        print(f"• Calendar access before: {status_names.get(before, before)}")
        if before == 4:
            print("  already granted ✓")
    except ImportError:
        print("• Calendar (EventKit status check unavailable, continuing)")

    if os.path.isfile(GET_CALENDARS_BIN):
        cmd = [GET_CALENDARS_BIN]  # compiled helper: carries the usage string
    elif os.path.isfile(GET_CALENDARS_SWIFT):
        cmd = ["swift", GET_CALENDARS_SWIFT]
    else:
        print(f"  helper not found in {CALENDAR_MCP} — is the MCP cloned?")
        return
    print("  running the calendar MCP's own Swift helper — click Allow if a "
          "prompt appears (waiting up to 120s)...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if r.returncode == 0 and out and "denied" not in out.lower():
        preview = out[:200].replace("\n", " ")
        print(f"  Calendar access working ✓ — helper returned: {preview}")
    else:
        print(f"  helper said: {(out or err)[:300]}")
        print("  No prompt is expected from a terminal on this macOS: grant "
              "manually via System Settings → Privacy & Security → Calendars "
              "(add your terminal app if a + button exists), or run "
              "grant_calendar_tcc.py from your terminal.")


def mail():
    print("• Mail automation: asking Mail for its account count via AppleScript...")
    r = subprocess.run(
        ["osascript", "-e", 'tell application "Mail" to count of accounts'],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode == 0:
        print(f"  Mail automation granted ✓ ({r.stdout.strip()} account(s) visible)")
    else:
        err = r.stderr.strip()
        print(f"  Mail said: {err}")
        print("  If you denied the prompt, re-enable in System Settings → "
              "Privacy & Security → Automation → your terminal → Mail")


if __name__ == "__main__":
    calendar()
    mail()
