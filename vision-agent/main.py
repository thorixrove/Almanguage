import os
from typing import Optional

from dotenv import load_dotenv

# Load Stream keys from the parent repo .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
# Local .env adds GOOGLE_API_KEY and can override any key
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from getstream.models import MemberRequest  # noqa: E402
from vision_agents.core import Agent, AgentLauncher, User, Runner  # noqa: E402
from vision_agents.core.instructions import Instructions  # noqa: E402
from vision_agents.core.agents.events import UserTranscriptEvent  # noqa: E402
from vision_agents.core.llm.events import LLMResponseFinalEvent  # noqa: E402
from vision_agents.plugins import getstream, gemini  # noqa: E402

AGENT_USER_ID = "ai-teacher"

LANGUAGE_NAMES: dict[str, str] = {
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "de": "German",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, energetic AI language teacher having a real voice conversation with a student. "
    "You operate in exactly two modes and NEVER mix them:\n"
    "TEACHING MODE: Say one word or phrase, its English meaning, and one pronunciation tip. "
    "End with a single question like 'Can you say that?' or 'Give it a try!'. "
    "Your turn is OVER at that question mark. Stop speaking. Output nothing else. "
    "Do NOT imagine what the student will say. Do NOT pre-write your reaction. Just stop.\n"
    "REACTING MODE: You have just received actual speech from the student in this turn. "
    "React to what they actually said — one sentence of praise or correction — "
    "then either ask them to try again or introduce the next word. Stop.\n"
    "ABSOLUTE RULES:\n"
    "- Never say 'Nice job', 'Perfect', 'Great', or any praise unless the student has "
    "ACTUALLY spoken in the current turn and you heard something from them.\n"
    "- Never continue past a question mark. Every question is a hard stop.\n"
    "- Never role-play the student's response or write what you imagine they said.\n"
    "- Keep every reply to one or two short sentences maximum.\n"
    "- Stay strictly within the current lesson's vocabulary."
)

def _require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise RuntimeError(f"Missing required environment variable: {var_name}")

def _language_name_from_call_id(call_id: str) -> Optional[str]:
    # call_id format: lesson-{langCode}-lesson-{n}-{userId}
    parts = call_id.split("-")
    if len(parts) >= 2 and parts[0] == "lesson":
        return LANGUAGE_NAMES.get(parts[1])
    return None


async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        # Gemini Live has built-in VAD-based turn detection enabled by default,
        # so no manual server_vad configuration is needed here (unlike OpenAI Realtime).
        llm=gemini.Realtime(),
        agent_user=User(name="AI Teacher", id=AGENT_USER_ID),
        instructions=DEFAULT_SYSTEM_PROMPT,
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Read lesson context packed into the call's custom data by the mobile app
    custom: dict = {}
    try:
        resp = await call.get()
        custom = resp.data.call.custom or {}
    except Exception as e:
        print(f"[agent] Warning: could not fetch call custom data: {e}")

    system_prompt  = custom.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    intro_message  = custom.get("intro_message")
    language_code  = custom.get("language") or ""
    lesson_title   = custom.get("lesson_title") or ""
    language_name  = LANGUAGE_NAMES.get(language_code) or _language_name_from_call_id(call_id) or "language"

    # Apply lesson-specific instructions before joining so the Realtime LLM receives them
    agent.instructions = Instructions(input_text=system_prompt)

    # Grant admin role + go live so the agent can publish audio
    try:
        await call.update_call_members(
            update_members=[MemberRequest(user_id=AGENT_USER_ID, role="admin")]
        )
    except Exception as e:
        print(f"[agent] Warning: could not set admin role: {e}")

    try:
        await call.go_live()
    except Exception as e:
        print(f"[agent] Warning: go_live failed (expected for default call type): {e}")

    # vision-agents 0.6.8 only exposes FINAL transcripts (no per-word delta streaming
    # like older versions did), so captions arrive as full sentences rather than
    # updating word-by-word while speech is in progress.
    async def on_transcript_event(event) -> None:
        if isinstance(event, UserTranscriptEvent):
            if event.text:
                try:
                    await agent.send_custom_event({
                        "type": "transcript_final",
                        "speaker": "user",
                        "text": event.text,
                    })
                except Exception as e:
                    print(f"[agent] send_custom_event error: {e}")

        elif isinstance(event, LLMResponseFinalEvent):
            if event.text:
                try:
                    await agent.send_custom_event({
                        "type": "transcript_final",
                        "speaker": "agent",
                        "text": event.text,
                    })
                except Exception as e:
                    print(f"[agent] send_custom_event error: {e}")

    agent.subscribe(on_transcript_event)

    async with agent.join(call):
        # Wait for the student to join (returns immediately if already present)
        await agent.wait_for_participant(timeout=60.0)

        if intro_message:
            context_parts = [f"A student just joined your {language_name} lesson"]
            if lesson_title:
                context_parts[0] += f" — '{lesson_title}'"
            context_parts[0] += "."
            context_parts.append(
                f"Deliver this greeting and NOTHING else: \"{intro_message}\" "
                f"After the greeting, ask the student one simple question to get them talking — "
                f"for example 'Are you ready to get started?' or 'Have you learned any {language_name} before?' "
                f"Then STOP and wait for the student's reply before teaching anything."
            )
            await agent.simple_response(" ".join(context_parts))
        else:
            await agent.simple_response(
                f"A student just joined your {language_name} lesson. "
                f"Greet them warmly and ask one short question — like 'Ready to learn some {language_name}?' "
                f"Then STOP and wait for their reply before you teach anything."
            )

        await agent.finish()


if __name__ == "__main__":
    _require_env("STREAM_API_KEY")
    _require_env("STREAM_API_SECRET")
    _require_env("GOOGLE_API_KEY")

    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()