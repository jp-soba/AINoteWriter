from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

from .config import AppConfig
from .models import PostWithContext


NO_NOTE_NEEDED = "NO NOTE NEEDED"
NOT_ENOUGH_EVIDENCE = "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE"


@dataclass
class AINoteDraft:
    note_text: str
    misleading_tags: list[str]


class AINoteGenerator:
    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    def _build_post_description(post_with_context: PostWithContext) -> str:
        lines: list[str] = []
        lines.append("[Target Post]")
        lines.append(post_with_context.post.text)

        if post_with_context.post.suggested_source_links:
            lines.append("\n[Suggested source links from requests]")
            lines.extend(post_with_context.post.suggested_source_links)

        if post_with_context.quoted_post is not None:
            lines.append("\n[Quoted Post]")
            lines.append(post_with_context.quoted_post.text)

        if post_with_context.in_reply_to_post is not None:
            lines.append("\n[In-reply-to Post]")
            lines.append(post_with_context.in_reply_to_post.text)

        return "\n".join(lines)

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        return re.findall(r"https?://[^\s)]+", text)

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        if not self.config.ai_api_key:
            return NO_NOTE_NEEDED

        base_url = self.config.ai_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.ai_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.config.ai_timeout_sec)
        if not resp.ok:
            raise RuntimeError(f"AI API request failed ({resp.status_code}): {resp.text}")
        body = resp.json()
        return body["choices"][0]["message"]["content"].strip()

    def _responses_completion(self, payload: dict[str, Any]) -> str:
        if not self.config.ai_api_key:
            return NO_NOTE_NEEDED

        base_url = self.config.ai_base_url.rstrip("/")
        url = f"{base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.config.ai_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.config.ai_timeout_sec)
        if not resp.ok:
            raise RuntimeError(f"AI API request failed ({resp.status_code}): {resp.text}")

        body = resp.json()
        # OpenAI Responses-compatible extraction
        output = body.get("output", [])
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if text:
                        return str(text).strip()
        raise RuntimeError("AI API response did not contain output text")

    def _get_prompt_for_live_search(self, post_with_context_description: str) -> str:
        return f"""以下は X の投稿です。投稿内の主張が誤解を招く可能性があるかを、公開情報で調査してください。

    要件:
    - 事実主張ごとに根拠URLを併記する
    - URLは本文にそのまま書く（「出典:」などの飾りは不要）
    - 不確かな情報は推測せず、確認できる情報のみ示す

    調査対象:
    {post_with_context_description}
    """.strip()

    def _get_prompt_for_note_writing(
        self,
        post_with_context_description: str,
        search_results: str,
    ) -> str:
        return f"""あなたは Community Notes の下書きを作成します。以下の投稿情報と調査結果を使って判断してください。

まず判定:
1) ノート不要なら "NO NOTE NEEDED." のみを返す
2) 必要性はあるが根拠不足なら "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE." のみを返す
3) 根拠が十分な場合のみ、ノート本文のみを返す

ノート本文ルール:
- URLを除いた本文は 280 文字以内を目安に簡潔に記述すること
- 体言止めなどの不自然な省略表現は避ける
- ノートは短い方が好まれる傾向にある。ただし、重要な情報は省略しないこと。
- 少なくとも1つのURLを本文に含める
- URLはそのまま記載（[Source] 等の装飾禁止）
- ハッシュタグ、絵文字、煽り表現は禁止
- 意見ではなく、検証可能な事実と文脈補足を中心に書く
- 冒頭に「Community Note:」等の前置きは付けない

判断方針:
- 迷う場合は書かない（NO NOTE NEEDED か NOT ENOUGH EVIDENCE を返す）
- 未来予測や主観のみの投稿には原則ノートを書かない
- 信頼性の高い一次情報・公的情報を優先する

投稿コンテキスト:
{post_with_context_description}

調査メモ:
```
{search_results}
```
""".strip()

    def _run_live_search(self, post_with_context_description: str) -> str:
        user_prompt = self._get_prompt_for_live_search(post_with_context_description)

        # xAI: migrate from deprecated `search_parameters` to Agent Tools style.
        if self.config.ai_provider.lower() == "xai":
            chat_payload: dict[str, Any] = {
                "model": self.config.ai_model,
                "temperature": 0.6,
                "messages": [
                    {
                        "role": "system",
                        "content": "あなたは慎重な調査アシスタントです。未確認情報を断定しません。",
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "tools": [
                    {
                        "type": "web_search",
                    }
                ],
                "tool_choice": "auto",
            }
            try:
                return self._chat_completion(chat_payload)
            except Exception:
                # Fallback: Responses API with built-in tool.
                responses_payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "input": user_prompt,
                    "tools": [
                        {
                            "type": "web_search",
                        }
                    ],
                    "tool_choice": "auto",
                }
                try:
                    return self._responses_completion(responses_payload)
                except Exception:
                    return ""

        payload: dict[str, Any] = {
            "model": self.config.ai_model,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "system",
                    "content": "あなたは慎重な調査アシスタントです。未確認情報を断定しません。",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        try:
            return self._chat_completion(payload)
        except Exception:
            return ""

    def generate_note(self, post_with_context: PostWithContext) -> AINoteDraft | None:
        if self.config.ai_provider.lower() in {"none", "off", "disabled"}:
            return None

        description = self._build_post_description(post_with_context)
        search_results = self._run_live_search(description)
        payload: dict[str, Any] = {
            "model": self.config.ai_model,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": "あなたはCommunity Notes向けの事実確認ライターです。",
                },
                {
                    "role": "user",
                    "content": self._get_prompt_for_note_writing(description, search_results),
                },
            ],
        }
        raw = self._chat_completion(payload)
        upper = raw.upper()
        if NO_NOTE_NEEDED in upper or NOT_ENOUGH_EVIDENCE in upper:
            return None

        urls = self._extract_urls(raw)
        if not urls:
            return None
        if "#" in raw:
            return None

        note_text = raw.strip()
        return AINoteDraft(
            note_text=note_text,
            misleading_tags=["missing_important_context"],
        )
