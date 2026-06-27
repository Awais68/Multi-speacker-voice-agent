"""
view_transcripts.py — CLI Transcript Viewer
Usage:
  python view_transcripts.py                    # all sessions
  python view_transcripts.py <session_id>       # specific session
  python view_transcripts.py search <query>     # search text
"""

import sys
import json
from transcript_manager import get_all_sessions, get_session_transcript, search_transcripts
from pathlib import Path

Path("transcripts").mkdir(parents=True, exist_ok=True)

def print_sessions():
    sessions = get_all_sessions()
    if not sessions:
        print("❌ No sessions found.")
        return
    print(f"\n{'='*60}")
    print(f"  📋 ALL SESSIONS ({len(sessions)} total)")
    print(f"{'='*60}")
    for s in sessions:
        status = "✅ Ended" if s["ended_at"] else "🔴 Active"
        print(f"""
  Session ID : {s['session_id']}
  Room       : {s['room_name']}
  Started    : {s['started_at']}
  Ended      : {s['ended_at'] or 'Still running'}
  Messages   : {s['total_messages']}
  Status     : {status}
  {'-'*50}""")


def print_transcript(session_id: str):
    entries = get_session_transcript(session_id)
    if not entries:
        print(f"❌ No transcript found for session: {session_id}")
        return
    print(f"\n{'='*60}")
    print(f"  📝 TRANSCRIPT — Session: {session_id}")
    print(f"{'='*60}\n")
    for e in entries:
        icon = "🤖" if e["role"] == "agent" else "👤"
        print(f"  {icon} [{e['timestamp'][11:19]}] {e['speaker']}")
        print(f"     {e['text']}")
        print()


def print_search(query: str):
    results = search_transcripts(query)
    if not results:
        print(f"❌ No results for: '{query}'")
        return
    print(f"\n{'='*60}")
    print(f"  🔍 SEARCH: '{query}' ({len(results)} results)")
    print(f"{'='*60}\n")
    for r in results:
        icon = "🤖" if r["role"] == "agent" else "👤"
        print(f"  {icon} {r['speaker']} | Session: {r['session_id']} | {r['timestamp'][0:19]}")
        print(f"     {r['text']}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print_sessions()
    elif args[0] == "search" and len(args) > 1:
        print_search(" ".join(args[1:]))
    else:
        print_transcript(args[0])
