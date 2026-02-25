from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Media:
    media_key: str
    media_type: str
    url: Optional[str] = None
    preview_image_url: Optional[str] = None


@dataclass
class Post:
    post_id: str
    author_id: str
    created_at: datetime
    text: str
    media: list[Media] = field(default_factory=list)
    suggested_source_links: list[str] = field(default_factory=list)


@dataclass
class PostWithContext:
    post: Post
    quoted_post: Optional[Post] = None
    in_reply_to_post: Optional[Post] = None


@dataclass
class ProposedNote:
    post_id: str
    note_text: str
    misleading_tags: list[str]
    trustworthy_sources: bool = True


@dataclass
class NoteProcessResult:
    post_id: str
    status: str
    reason: Optional[str] = None
    generated_note: Optional[str] = None
    claim_opinion_score: Optional[float] = None
    submission_response: Optional[dict] = None


@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    test_mode: bool
    submit_notes: bool
    evaluate_before_submit: bool
    num_posts_requested: int
    num_posts_fetched: int
    results: list[NoteProcessResult]
