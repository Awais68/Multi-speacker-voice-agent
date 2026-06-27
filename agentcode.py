"""
agentcode.py — Zoya Multi-Speaker Voice Agent
With full transcript saving (memory + SQLite + JSON)
"""

import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import openai, cartesia
from transcript_manager import TranscriptManager

logger = logging.getLogger("zoya-agent")
load_dotenv()


# ─── Speaker Tracker ──────────────────────────────────
class SpeakerTracker:
    def __init__(self):
        self.active_speakers: dict[str, bool] = {}
        self.speaker_names: dict[str, str] = {}
        self.last_speaker: str | None = None

    def register(self, identity: str, name: str) -> None:
        self.speaker_names[identity] = name or identity
        logger.info(f"👤 Registered: {identity} → {name}")

    def unregister(self, identity: str) -> None:
        self.speaker_names.pop(identity, None)
        self.active_speakers.pop(identity, None)
        if self.last_speaker == identity:
            self.last_speaker = None
        logger.info(f"👋 Unregistered: {identity}")

    def speaker_active(self, identity: str) -> None:
        self.active_speakers[identity] = True
        self.last_speaker = identity

    def speaker_stopped(self, identity: str) -> None:
        self.active_speakers[identity] = False

    def get_name(self, identity: str) -> str:
        return self.speaker_names.get(identity, identity)

    def get_current_speaker(self) -> str | None:
        return self.last_speaker

    def multiple_speaking(self) -> bool:
        return sum(self.active_speakers.values()) > 1

    def get_active_names(self) -> list[str]:
        return [
            self.get_name(pid)
            for pid, active in self.active_speakers.items()
            if active
        ]

    def get_context(self) -> str:
        all_speakers = list(self.speaker_names.values())
        active = self.get_active_names()
        last = self.get_name(self.last_speaker) if self.last_speaker else "Unknown"
        return (
            f"Room participants: {', '.join(all_speakers) or 'None'}\n"
            f"Currently speaking: {', '.join(active) or 'No one'}\n"
            f"Last speaker: {last}"
        )


tracker = SpeakerTracker()


# ─── Agent ────────────────────────────────────────────
class ZoyaAgent(Agent):
    def __init__(self, transcript: TranscriptManager) -> None:
        self.transcript = transcript
        super().__init__(
            instructions=self._build_instructions(),
        )

    def _build_instructions(self) -> str:
        # Recent conversation history inject karo LLM context mein
        history = self.transcript.get_formatted(last_n=10) if hasattr(self, 'transcript') else ""

        return f"""You are Zoya, a voice assistant for multi-speaker conversations.

{tracker.get_context()}

Recent Conversation:
{history}

Rules:
- Only respond when someone directly addresses you as "Zoya"
- Always use the speaker's name in your response
- If multiple people talking: say "Could one person speak at a time?"
- If unsure who spoke: ask "Were you speaking to me?"
- Keep responses under 3 sentences
- Never interrupt anyone
"""

    async def on_enter(self):
        names = list(tracker.speaker_names.values())
        greeting = (
            f"Hello {', '.join(names)}! I'm Zoya. Address me by name when you need me."
            if names
            else "Hello! I'm Zoya. Address me by name when you need me."
        )
        # Agent greeting bhi transcript mein save karo
        await self.transcript.add(
            speaker="Zoya",
            text=greeting,
            role="agent",
        )
        await self.session.generate_reply(
            instructions=greeting,
            allow_interruptions=True,
        )


# ─── Server ───────────────────────────────────────────
server = AgentServer()


def prewarm(proc: JobProcess):
    pass


server.setup_fnc = prewarm


@server.rtc_session(agent_name="zoya-agent")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # ── Unique session ID generate karo ──
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name or "default-room"

    # ── Transcript Manager init ──
    transcript = TranscriptManager(
        session_id=session_id,
        room_name=room_name,
    )

    # ── Participant events ──
    @ctx.room.on("participant_connected")
    def on_join(participant: rtc.RemoteParticipant):
        tracker.register(participant.identity, participant.name or participant.identity)
        logger.info(f"✅ Joined: {participant.identity}")

    @ctx.room.on("participant_disconnected")
    def on_leave(participant: rtc.RemoteParticipant):
        tracker.unregister(participant.identity)

    # Already-joined participants
    for p in ctx.room.remote_participants.values():
        tracker.register(p.identity, p.name or p.identity)

    # ── Speaker tracking ──
    @ctx.room.on("active_speakers_changed")
    def on_speakers_changed(speakers: list[rtc.Participant]):
        active_ids = {s.identity for s in speakers}
        for pid in list(tracker.active_speakers.keys()):
            if pid not in active_ids:
                tracker.speaker_stopped(pid)
        for s in speakers:
            tracker.speaker_active(s.identity)
        if tracker.multiple_speaking():
            logger.warning(f"⚠️ Multiple speaking: {tracker.get_active_names()}")

    # ── Session ──
    session = AgentSession(
        stt=openai.STT(
            model="whisper-large-v3-turbo",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        ),
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice="b53fd0c8-834a-45a6-862a-d2f6bc41a2bc",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector()
        ),
    )

    # ── USER transcript save ──────────────────────────
    @session.on("user_speech_committed")
    def on_user_speech(msg):
        speaker_id = tracker.get_current_speaker()
        speaker_name = tracker.get_name(speaker_id) if speaker_id else "Unknown"

        asyncio.ensure_future(
            transcript.add(
                speaker=speaker_name,
                text=msg.content,
                role="user",
            )
        )
        logger.info(f"🗣️  [{speaker_name}]: {msg.content}")

    # ── AGENT transcript save ─────────────────────────
    @session.on("agent_speech_committed")
    def on_agent_speech(msg):
        asyncio.ensure_future(
            transcript.add(
                speaker="Zoya",
                text=msg.content,
                role="agent",
            )
        )
        logger.info(f"🤖 [Zoya]: {msg.content}")

    # ── Dynamic instructions refresh ──────────────────
    async def refresh_context():
        while True:
            await asyncio.sleep(3)
            try:
                if session.agent:
                    session.agent.instructions = ZoyaAgent(transcript)._build_instructions()
            except Exception:
                pass

    asyncio.create_task(refresh_context())

    # ── Session end — transcript save karo ────────────
    @ctx.room.on("disconnected")
    def on_disconnect():
        asyncio.ensure_future(on_session_end())

    async def on_session_end():
        stats = transcript.get_stats()
        logger.info(f"📊 Session stats: {stats}")
        await transcript.end_session()

    await session.start(
        agent=ZoyaAgent(transcript),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
