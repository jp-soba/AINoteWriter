"""MCP server providing note utility tools for Community Notes."""

from __future__ import annotations

import re

from fastmcp import FastMCP

mcp = FastMCP("note-tools")


def _count_note_chars(text: str) -> int:
    """Count characters in a note with URL collapsing.

    Each URL (https?://...) counts as 1 character regardless of length.
    Every other character (Japanese, English, spaces, newlines) counts as 1.
    """
    collapsed = re.sub(r"https?://\S+", "X", text)
    return len(collapsed)


@mcp.tool()
def count_note_chars(note_text: str) -> dict:
    """Count characters in a Community Note.

    URLs are collapsed to 1 character each. All other characters
    (including Japanese, English, spaces, newlines) count as 1.
    The limit for Community Notes is 280 characters.
    """
    char_count = _count_note_chars(note_text)
    return {
        "char_count": char_count,
        "limit": 280,
        "over": char_count > 280,
        "remaining": 280 - char_count,
    }


if __name__ == "__main__":
    mcp.run()