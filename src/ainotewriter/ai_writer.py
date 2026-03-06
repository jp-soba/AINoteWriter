from __future__ import annotations

import asyncio
import importlib
import inspect
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

import logging

from .config import AppConfig
from .models import PostWithContext

logger = logging.getLogger(__name__)

NO_NOTE_NEEDED = "NO NOTE NEEDED"
NOT_ENOUGH_EVIDENCE = "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE"


@dataclass
class AINoteDraft:
    note_text: str
    misleading_tags: list[str]


@dataclass
class NoteGenerationResult:
    draft: AINoteDraft | None
    reason: str


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

    @staticmethod
    def _extract_text_recursive(value: Any) -> list[str]:
        texts: list[str] = []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                texts.append(stripped)
            return texts

        if isinstance(value, dict):
            for key in ("text", "content", "message", "result", "value", "output"):
                if key in value:
                    texts.extend(AINoteGenerator._extract_text_recursive(value[key]))
            return texts

        if isinstance(value, list):
            for item in value:
                texts.extend(AINoteGenerator._extract_text_recursive(item))
            return texts

        return texts

    @staticmethod
    def _build_kwargs_for_signature(func: Any, prompt: str, system_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return kwargs

        params = sig.parameters
        if "prompt" in params:
            kwargs["prompt"] = prompt
        elif "query" in params:
            kwargs["query"] = prompt
        elif "input" in params:
            kwargs["input"] = prompt

        if "system_prompt" in params:
            kwargs["system_prompt"] = system_prompt
        elif "system" in params:
            kwargs["system"] = system_prompt

        return kwargs

    async def _run_claude_sdk_query_async(self, prompt: str, system_prompt: str) -> str:
        sdk = importlib.import_module("claude_code_sdk")
        query = getattr(sdk, "query", None)
        if query is None:
            raise RuntimeError("claude_code_sdk.query is not available")

        kwargs = self._build_kwargs_for_signature(query, prompt, system_prompt)

        options_cls = getattr(sdk, "ClaudeCodeOptions", None)
        if options_cls is not None:
            try:
                option_sig = inspect.signature(options_cls)
                option_kwargs: dict[str, Any] = {}
                if "max_turns" in option_sig.parameters:
                    option_kwargs["max_turns"] = self.config.claude_max_turns
                if "system_prompt" in option_sig.parameters:
                    option_kwargs["system_prompt"] = system_prompt
                if option_kwargs and "options" in inspect.signature(query).parameters:
                    kwargs["options"] = options_cls(**option_kwargs)
            except (TypeError, ValueError):
                pass

        result = query(**kwargs) if kwargs else query(prompt)

        if inspect.isawaitable(result):
            result = await result

        chunks: list[str] = []
        if hasattr(result, "__aiter__"):
            async for event in result:
                chunks.extend(self._extract_text_recursive(event))
        else:
            chunks.extend(self._extract_text_recursive(result))

        merged = "\n".join(chunk for chunk in chunks if chunk).strip()
        if not merged:
            raise RuntimeError("Claude Agent SDK returned empty response")
        return merged

    def _run_claude_cli_prompt(
        self, prompt: str, system_prompt: str, *, allow_web_tools: bool = False
    ) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}".strip()
        cmd = [self.config.claude_cli_path, "--print"]
        if allow_web_tools:
            cmd += ["--allowedTools", "WebSearch,WebFetch"]
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=self.config.ai_timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            raise RuntimeError(
                f"Claude CLI request failed (exit={proc.returncode}): {stderr or 'unknown error'}"
            )
        text = (proc.stdout or "").strip()
        if not text:
            raise RuntimeError("Claude CLI returned empty response")
        return text

    def _claude_completion(
        self, prompt: str, system_prompt: str, *, allow_web_tools: bool = False
    ) -> str:
        sdk_error: Exception | None = None
        try:
            return asyncio.run(self._run_claude_sdk_query_async(prompt, system_prompt))
        except Exception as ex:
            sdk_error = ex

        if self.config.claude_use_cli_fallback:
            return self._run_claude_cli_prompt(
                prompt, system_prompt, allow_web_tools=allow_web_tools
            )

        raise RuntimeError(f"Claude Agent SDK request failed: {sdk_error}")

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
- 文体は丁寧語で統一すること
- ノートは短い方が好まれる傾向にある。ただし、重要な情報は省略しないこと。
- 少なくとも1つのURLを本文に含める
- URLはそのまま記載（[Source]やカッコで囲む等の装飾禁止）。URL以外のテキストとは、改行等で区切ること
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
        provider = self.config.ai_provider.lower()

        if provider in {"claude", "claude_agent", "claude-agent"}:
            try:
                return self._claude_completion(
                    prompt=user_prompt,
                    system_prompt="あなたは慎重な調査アシスタントです。公開情報のみを使い、根拠URLを明示してください。",
                    allow_web_tools=True,
                )
            except Exception as ex:
                logger.warning("Live search failed (claude): %s", ex)
                return ""

        if provider == "xai":
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
            except Exception as ex:
                logger.warning("Live search failed (xai chat): %s", ex)
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
                except Exception as ex:
                    logger.warning("Live search failed (xai responses): %s", ex)
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
        except Exception as ex:
            logger.warning("Live search failed: %s", ex)
            return ""

    def generate_note(self, post_with_context: PostWithContext) -> NoteGenerationResult:
        provider = self.config.ai_provider.lower()
        if provider in {"none", "off", "disabled"}:
            return NoteGenerationResult(draft=None, reason="disabled")

        description = self._build_post_description(post_with_context)
        search_results = self._run_live_search(description)
        logger.info("Live search results:\n%s", search_results if search_results else "(empty)")

        note_prompt = self._get_prompt_for_note_writing(description, search_results)

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=note_prompt,
                    system_prompt="あなたはCommunity Notes向けの事実確認ライターです。推測はせず、検証可能な事実のみを使ってください。",
                )
            else:
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
                            "content": note_prompt,
                        },
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Note generation failed: %s", ex)
            return NoteGenerationResult(draft=None, reason=f"ai_error ({ex})")

        logger.info("AI raw response:\n%s", raw)

        upper = raw.upper()
        if NO_NOTE_NEEDED in upper:
            return NoteGenerationResult(draft=None, reason="no_note_needed")
        if NOT_ENOUGH_EVIDENCE in upper:
            return NoteGenerationResult(draft=None, reason="not_enough_evidence")

        urls = self._extract_urls(raw)
        if not urls:
            return NoteGenerationResult(draft=None, reason="no_url")
        if "#" in raw:
            return NoteGenerationResult(draft=None, reason="hashtag")

        note_text = raw.strip()
        draft = AINoteDraft(
            note_text=note_text,
            misleading_tags=["missing_important_context"],
        )
        return NoteGenerationResult(draft=draft, reason="ok")