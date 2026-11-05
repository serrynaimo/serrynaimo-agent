"""Is an utterance that passed the voice gate meant for the assistant?

The speaker and wake gates know WHO spoke, not to whom. With other people
around, the enrolled user talks to them too ("honey, dinner?", "yes, I locked
the car"), and inside the wake window those lines reach the bot as requests.
So the classifier endpoint on the LM Studio host (a StartLux Decision model
that answers typed questions about a JSON state) gets one yes/no question
over the last few chat turns plus the new message.

The wording was picked on addressee_eval.py's cases, and small changes move
the answers a lot: rerun it after touching QUESTION or state().
"""

import os
import re

import aiohttp
from loguru import logger

QUESTION = (
    "Is {user}'s new message meant for {user}'s AI assistant {agent}, rather than for another "
    "person? Questions and requests an assistant handles count as yes even when they change "
    "topic; side talk with people in the room or on the phone is no."
)


def state(
    trail: list[tuple[str, str]],
    side_talk: list[str],
    new_message: str,
    *,
    user: str,
    agent: str,
    secs_since_agent: float | None,
    others_present: bool,
) -> dict:
    """The classifier's view: chat trail, recent side talk, and the new message."""
    lines = [f"{who}: {text}" for who, text in trail]
    lines += [f"{user} (to someone else): {text}" for text in side_talk]
    s: dict = {"chat_trail": lines}
    if secs_since_agent is not None:
        key = re.sub(r"\W+", "_", agent.lower())
        s[f"seconds_since_{key}_spoke"] = round(secs_since_agent)
    s["other_people_present"] = "yes" if others_present else "no"
    s["new_message"] = f"{user}: {new_message}"
    return s


async def p_meant_for_agent(
    s: dict, *, user: str, agent: str, timeout: float = 1.5
) -> float | None:
    """Probability the new message is meant for the assistant, or None when
    the classifier is off, slow or failing (callers then let it through).

    ADDRESSEE_CLASSIFIER_URL overrides the endpoint (default:
    LMSTUDIO_BASE_URL's /classifier).
    """
    url = os.getenv("ADDRESSEE_CLASSIFIER_URL") or (
        f"{os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')}/classifier"
    )
    question = {"type": "noul", "instructions": QUESTION.format(user=user, agent=agent)}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                # xhigh: the reply waits on this, unlike the background mood check.
                json={"state": s, "questions": {"addressed": question}, "priority": "xhigh"},
                headers={"Authorization": f"Bearer {os.getenv('LMSTUDIO_API_KEY', 'lm-studio')}"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    logger.warning(f"Addressee classifier {response.status}: {str(data)[:200]}")
                    return None
        return float(data["answers"]["addressed"]["noul"])
    except Exception as exc:  # noqa: BLE001 — fail open: never lock the user out
        logger.warning(f"Addressee classifier failed (letting the utterance through): {exc!r}")
        return None
