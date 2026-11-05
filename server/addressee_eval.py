"""Check the addressee classifier on hand-written cases.

Each case is a short chat trail with the assistant, whether other people were
heard recently, the user's recent side talk, and a new line that is either
meant for the assistant or for someone else. Reports how many real requests
get through and how much side talk is caught at ADDRESSEE_MIN_YES. Rerun it
after changing addressee.QUESTION or addressee.state(); "tuning" cases shaped
the wording, "held-out" ones did not.

usage: uv run addressee_eval.py
"""

import asyncio
import os
import statistics
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import addressee  # noqa: E402

USER = os.getenv("USER_NAME_SHORT", "Alex")
AGENT = os.getenv("AGENT_NAME", "Serrynaimo")
MIN_YES = float(os.getenv("ADDRESSEE_MIN_YES", "0.5"))

A, O = "assistant", "someone_else"
ANNA = [
    (USER, "Who is Anna?"),
    (AGENT, "Anna's your oldest sister, and she turns forty next week."),
]
WEATHER = [
    (USER, "What's the weather tomorrow?"),
    (AGENT, "Sunny and 31 degrees, with a chance of a storm in the late afternoon."),
]
TABLE = [
    (USER, "Book a table at the Italian place for Friday."),
    (AGENT, "Sure. What time should I book it for?"),
]
CAL = [
    (USER, "Add the dentist on Monday."),
    (AGENT, "Done. Do you want a reminder the evening before?"),
]

# (expected, trail, other people heard, side talk, seconds since the assistant spoke, new line)
TUNING = [
    (A, ANNA, False, [], 3, "What should I get her?"),
    (A, ANNA, True, [], 3, "What should I get her?"),
    (A, WEATHER, True, [], 4, "And what about Saturday?"),
    (A, TABLE, True, [], 2, "I think around seven."),
    (A, CAL, True, [], 2, "Yes please."),
    (A, WEATHER, False, [], 40, "Remind me to call the bank at three."),
    (A, WEATHER, True, [], 25, "Can you turn the volume down a bit?"),
    (A, ANNA, True, [], 5, "No, I meant Anna from the office, not my sister."),
    (
        A,
        WEATHER,
        True,
        ["Do you want to go to the beach on Saturday?"],
        30,
        "What's the tide like on Saturday morning?",
    ),
    (A, ANNA, False, [], 6, "Thanks, that's helpful."),
    (
        A,
        CAL,
        True,
        ["Yeah, I'll pick up the kids."],
        15,
        "Actually, also add a reminder for the kids' pickup at four.",
    ),
    (O, WEATHER, True, [], 20, "Honey, do you want to go out for dinner tonight?"),
    (
        O,
        ANNA,
        True,
        ["Did you get her a present yet?"],
        30,
        "Not yet, I'll go to the mall tomorrow.",
    ),
    (O, TABLE, True, ["Are you coming to the party?"], 45, "I think around seven."),
    (O, CAL, True, ["Where are my keys?"], 30, "No, the blue one, it's in the top drawer."),
    (O, WEATHER, False, [], 60, "Hi Mum, yes I'm home now, we can talk."),
    (O, ANNA, True, [], 25, "Hang on, let me just finish this email first."),
    (
        O,
        WEATHER,
        True,
        ["We should leave by six."],
        20,
        "Yeah, otherwise we'll get stuck in traffic again.",
    ),
    (O, CAL, True, [], 35, "Can you pass me the charger, it's next to you?"),
    (O, TABLE, True, ["The kids want pizza."], 50, "Okay, then let's just order in."),
    (O, ANNA, False, [], 90, "Yes, yes, I'll send you the slides after the call."),
    (O, WEATHER, True, ["Did you lock the car?"], 15, "Yes, I did, I checked twice."),
]
HELD_OUT = [
    (A, WEATHER, True, [], 3, "Will I need an umbrella?"),
    (A, ANNA, True, [], 4, "When exactly is her party?"),
    (A, CAL, False, [], 5, "No, don't bother."),
    (A, TABLE, True, ["Shall we go out on Friday?"], 30, "Book it for eight then."),
    (A, WEATHER, True, [], 15, "Play some jazz."),
    (A, ANNA, True, [], 8, "Hmm, okay, what else is on tomorrow?"),
    (O, WEATHER, True, ["Should we take the umbrella?"], 12, "Yeah, take the big one."),
    (O, ANNA, True, [], 20, "Mia, can you get the cake from the fridge?"),
    (O, CAL, True, ["What time is the dentist?"], 25, "Monday at ten, I think."),
    (O, TABLE, False, [], 40, "Sorry, the line is bad, I'll call you back in five minutes."),
    (O, WEATHER, True, [], 30, "Did you feed the cat?"),
    (O, ANNA, True, ["Is she coming with Paul?"], 18, "No, Paul's in Germany this week."),
]


async def main():
    latencies = []
    for label, cases in (("tuning", TUNING), ("held-out", HELD_OUT)):
        kept = caught = n_requests = n_side = 0
        misses = []
        for expected, trail, others, side, secs, line in cases:
            s = addressee.state(
                trail,
                side,
                line,
                user=USER,
                agent=AGENT,
                secs_since_agent=secs,
                others_present=others,
            )
            t = time.perf_counter()
            p = await addressee.p_meant_for_agent(s, user=USER, agent=AGENT, timeout=10)
            latencies.append(time.perf_counter() - t)
            if p is None:
                print(f"  classifier failed on: {line}")
                continue
            if expected == A:
                n_requests += 1
                kept += p >= MIN_YES
            else:
                n_side += 1
                caught += p < MIN_YES
            if (p >= MIN_YES) != (expected == A):
                misses.append(
                    f"    p {p:.2f}  {'request ' if expected == A else 'side talk'}  others={'yes' if others else 'no '}  {line}"
                )
        print(f"{label}: requests through {kept}/{n_requests}, side talk caught {caught}/{n_side}")
        print("\n".join(misses))
    print(
        f"latency median {1000 * statistics.median(latencies):.0f} ms, "
        f"max {1000 * max(latencies):.0f} ms (repeat runs hit the server's cache)"
    )


if __name__ == "__main__":
    asyncio.run(main())
