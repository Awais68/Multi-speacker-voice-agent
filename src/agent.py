import logging
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
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import openai, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-multi-speaker-Zoya")

load_dotenv(".env.local")


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a voice assistant namely Zoya designed for multi-speaker conversations. 

Your behavior:
- When multiple people are speaking, wait until one person finishes before responding
- Address only the person who is directly asking you a question
- If you are unsure who is speaking to you, ask \"Were you speaking to me?\"
- Keep responses short and conversational — under 3 sentences
- Do not interrupt speakers

You can answer general questions, assist with tasks, and hold natural conversations.""",
            tools=[EndCallTool(
                extra_description="""""",
                end_instructions="""Thank the user for their time and say goodbye.""",
                delete_room=False,
            )],
        )
    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""Hello! I'm Zoya. Go ahead and speak — I'll respond when you address me directly.""",
            allow_interruptions=True,
        )


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="multi-speaker-Zoya")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-multi", language="multi"),
        llm=inference.LLM(
            model="openai/gpt-4o",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en-US"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.NC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
