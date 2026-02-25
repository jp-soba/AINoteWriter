from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import requests
from requests_oauthlib import OAuth1

from .config import AppConfig
from .models import Media, Post, PostWithContext, ProposedNote


class XCommunityNotesClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.validate_x_auth()
        self.auth = OAuth1(
            client_key=config.x_api_key,
            client_secret=config.x_api_key_secret,
            resource_owner_key=config.x_access_token,
            resource_owner_secret=config.x_access_token_secret,
        )

    def _url(self, path: str) -> str:
        return f"{self.config.x_api_base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _raise_for_error(resp: requests.Response) -> None:
        if resp.ok:
            return
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f"X API request failed ({resp.status_code}): {payload}")

    @staticmethod
    def _parse_post(item: dict[str, Any], media_by_key: dict[str, dict[str, Any]]) -> Post:
        media_keys = item.get("attachments", {}).get("media_keys", [])
        media = []
        for key in media_keys:
            if key in media_by_key:
                m = media_by_key[key]
                media.append(
                    Media(
                        media_key=m.get("media_key", key),
                        media_type=m.get("type", "unknown"),
                        url=m.get("url"),
                        preview_image_url=m.get("preview_image_url"),
                    )
                )

        text = item.get("note_tweet", {}).get("text") or item.get("text", "")
        links = [
            x.get("url")
            for x in item.get("suggested_source_links_with_counts", [])
            if x.get("url")
        ]
        created_at = item.get("created_at") or datetime.utcnow().isoformat() + "Z"
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return Post(
            post_id=item["id"],
            author_id=item.get("author_id", ""),
            created_at=created_at,
            text=text,
            media=media,
            suggested_source_links=links,
        )

    def get_posts_eligible_for_notes(
        self,
        max_results: int = 5,
        test_mode: bool = True,
        post_selection: Optional[str] = None,
    ) -> list[PostWithContext]:
        params = {
            "test_mode": str(test_mode).lower(),
            "max_results": max_results,
            "tweet.fields": "author_id,created_at,referenced_tweets,media_metadata,note_tweet,suggested_source_links_with_counts",
            "expansions": "attachments.media_keys,referenced_tweets.id,referenced_tweets.id.attachments.media_keys",
            "media.fields": "alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,variants",
        }
        if post_selection:
            params["post_selection"] = post_selection

        resp = requests.get(self._url("notes/search/posts_eligible_for_notes"), params=params, auth=self.auth, timeout=30)
        self._raise_for_error(resp)
        payload = resp.json()

        includes = payload.get("includes", {})
        media_by_key = {m["media_key"]: m for m in includes.get("media", []) if "media_key" in m}

        data_items = payload.get("data", [])
        referenced_items = includes.get("tweets", [])
        referenced_by_id = {
            item["id"]: self._parse_post(item, media_by_key)
            for item in referenced_items
            if "id" in item
        }

        out: list[PostWithContext] = []
        for item in data_items:
            post = self._parse_post(item, media_by_key)
            quoted_post = None
            in_reply_to_post = None
            for ref in item.get("referenced_tweets", []):
                ref_type = ref.get("type")
                ref_id = ref.get("id")
                if not ref_id:
                    continue
                if ref_type == "quoted":
                    quoted_post = referenced_by_id.get(ref_id)
                elif ref_type == "replied_to":
                    in_reply_to_post = referenced_by_id.get(ref_id)
            out.append(PostWithContext(post=post, quoted_post=quoted_post, in_reply_to_post=in_reply_to_post))

        return out

    def evaluate_note(self, post_id: str, note_text: str) -> dict[str, Any]:
        body = {"post_id": post_id, "note_text": note_text}
        resp = requests.post(self._url("evaluate_note"), json=body, auth=self.auth, timeout=30)
        self._raise_for_error(resp)
        return resp.json()

    def submit_note(self, note: ProposedNote, test_mode: bool = True) -> dict[str, Any]:
        body = {
            "test_mode": test_mode,
            "post_id": note.post_id,
            "info": {
                "text": note.note_text,
                "classification": "misinformed_or_potentially_misleading",
                "misleading_tags": note.misleading_tags,
                "trustworthy_sources": note.trustworthy_sources,
            },
        }
        resp = requests.post(self._url("notes"), json=body, auth=self.auth, timeout=30)
        self._raise_for_error(resp)
        return resp.json()

    def get_notes_written(self, max_results: int = 20, test_mode: bool = True) -> dict[str, Any]:
        params = {
            "max_results": max_results,
            "test_mode": test_mode,
        }
        resp = requests.get(self._url("notes/search/notes_written"), params=params, auth=self.auth, timeout=30)
        self._raise_for_error(resp)
        return resp.json()
