import asyncio
import json
from ollama import AsyncClient

async def run_llm_test():
    client = AsyncClient()
    
    test_cases = [
        {"artist": "The Weeknd", "title": "Blinding Lights (Official Music Video) [4K]"},
        {"artist": "Lex Fridman", "title": "Episode 312: Interview with John Carmack"},
        {"artist": "Spotify", "title": "Advertisement - Upgrade to Premium"}
    ]

    for case in test_cases:
        print(f"\n--- Testing: {case['artist']} - {case['title']} ---")
        
        prompt = f"""Classify this media item and clean its title.

Rules:
1. "type" is "SONG" for music tracks, "OTHER" for podcasts, interviews, ads, vlogs.
2. "clean_title" MUST have every promotional tag removed. Delete any
   parenthesised or bracketed segment such as (Official Music Video),
   (Official Audio), (Lyrics), [4K], [HD], and any trailing "| Artist Name".
   Keep genuine title parts like "(feat. X)" or "(Remix)".
3. "clean_artist" is the artist name only, with no channel suffixes
   such as "VEVO" or "- Topic".

Examples:
Input: artist="Dua Lipa", title="Levitating (Official Music Video) [HD]"
Output: {{"type": "SONG", "clean_artist": "Dua Lipa", "clean_title": "Levitating"}}
Input: artist="Joe Rogan", title="JRE #1500 - Elon Musk"
Output: {{"type": "OTHER", "clean_artist": "Joe Rogan", "clean_title": "JRE #1500 - Elon Musk"}}

Now do the same for:
artist="{case['artist']}"
title="{case['title']}"

Respond ONLY with a JSON object of the form:
{{"type": "SONG" or "OTHER", "clean_artist": "...", "clean_title": "..."}}"""

        response = await client.chat(
            model='LiquidAI/lfm2.5-1.2b-instruct:latest',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0}  # deterministic output for a classifier
        )

        print(json.dumps(json.loads(response.message.content), indent=2))

asyncio.run(run_llm_test())