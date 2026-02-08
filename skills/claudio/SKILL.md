---
name: claudio
description: Voice output for Claude Code. Speaks text wrapped in say tags while the claudio proxy is running.
user-invocable: false
---

# Voice Output

You have text-to-speech enabled. ALWAYS wrap your text in `<say>` tags - this is how you speak to the user.

## Critical Rules
- Wrap ALL your spoken text in `<say>` tags — if it's not in tags, the user won't hear it
- Speak most of your response: explanations, reasoning, answers, summaries, and narration should all be spoken
- NEVER read code blocks, terminal output, file paths, URLs, or markdown formatting aloud — instead, narrate what they contain (e.g. "Here's the updated function" or "The test output shows three failures")
- Break long responses into multiple `<say>` blocks, placing them before and after code/formatting blocks
- Keep the user informed as you work — narrate tool calls, searches, and findings

## Syntax
```xml
<say>Your normal spoken text goes here.</say>
<say speed="fast">Quick updates like this.</say>
```

## Example Response
```
<say>I found the issue. Here's a fix using a caching decorator:</say>

```python
@cache(ttl=300)
def get_users():
    return db.query("SELECT * FROM users")
```

<say>This caches results for 5 minutes. Want me to add error handling too?</say>
```

## Narrate Your Work

Keep the user informed while tools run and code executes:

```xml
<say>Let me look at that file.</say>
<say>Found it. Fixing the off-by-one error now.</say>
<say>Running the tests.</say>
<say>All passing. Here's what I changed:</say>
```

Remember: No `<say>` tags = silence. Always speak to your user!

## Changing Voice Settings

If the user asks to change voice, speed, or mute/unmute, run these via bash:

**Change voice:**
```bash
claudio session voice <session_id> <voice_name>
```

**Change persona** (sets voice, speed, and personality together):
```bash
claudio persona set <persona_id>              # narrator, alert, coder
claudio persona set <persona_id> --session <session_id>
```

**Mute / unmute:**
```bash
claudio session mute <session_id>
claudio session unmute <session_id>
```

**Check current settings:**
```bash
claudio persona get                           # shows voice, speed, priority
```

**List available voices and personas:**
```bash
claudio voices
claudio persona list
```

Use `default` as the session ID when the user doesn't specify one.

Speed can also be overridden per-utterance with the `speed` attribute on say tags:
```xml
<say speed="1.5">This is spoken faster.</say>
<say speed="0.7">This is spoken slower.</say>
```
