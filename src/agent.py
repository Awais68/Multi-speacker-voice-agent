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
    # ✅ CHANGE 1: room_io REMOVE karo — noise_cancellation nahi use karenge
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import openai, cartesia
# ✅ CHANGE 2: silero aur noise_cancellation import REMOVE

logger = logging.getLogger("agent-multi-speaker-Zoya")
load_dotenv(".env.local")

class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a voice assistant namely Zoya designed for multi-speaker conversations. 
Your behavior:
- When multiple people are speaking, wait until one person finishes before responding
- Address only the person who is directly asking you a question
- If you are unsure who is speaking to you, ask "Were you speaking to me?"
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
    # ✅ CHANGE 3: silero.VAD.load() REMOVE — AgentSession bundled VAD use karta hai
    pass

server.prewarm_fnc = prewarm  # setup_fnc → prewarm_fnc

@server.rtc_session(agent_name="multi-speaker-Zoya")
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
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
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
        # vad= REMOVE — bundled hai
        preemptive_generation=True,
    )

    # ✅ CHANGE 4: room_options REMOVE — noise_cancellation nahi hai
    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)
