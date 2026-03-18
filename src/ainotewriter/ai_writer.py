from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

import logging

from .config import AppConfig
from .models import PostWithContext

logger = logging.getLogger(__name__)


def _load_prompts() -> dict:
    path = Path(__file__).parent / "prompts.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_PROMPTS: dict = _load_prompts()

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
    rewrite_count: int = 0


class AINoteGenerator:
    def __init__(self, config: AppConfig, feed_lang: str | None = None):
        self.config = config
        self.feed_lang = feed_lang
        self.lang = "ja" if feed_lang == "ja" else "en"

    def _p(self, category: str, key: str) -> str:
        return _PROMPTS[category][key][self.lang]

    # ── Post description / media helpers ──

    def _build_post_description(self, post_with_context: PostWithContext) -> str:
        lines: list[str] = []
        lines.append("[Target Post]")
        lines.append(post_with_context.post.text)

        if post_with_context.post.media:
            photo_count = sum(1 for m in post_with_context.post.media if m.media_type == "photo")
            video_count = sum(1 for m in post_with_context.post.media if m.media_type != "photo")
            if photo_count:
                lines.append(self._p("ui_strings", "photo_attachment").format(count=photo_count))
            if video_count:
                lines.append(self._p("ui_strings", "video_attachment"))

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
    def _get_image_urls(post_with_context: PostWithContext) -> list[str]:
        urls: list[str] = []
        for post in [post_with_context.post, post_with_context.quoted_post, post_with_context.in_reply_to_post]:
            if post is None:
                continue
            for media in post.media:
                if media.media_type == "photo":
                    url = media.url or media.preview_image_url
                    if url:
                        urls.append(url)
        return urls

    @staticmethod
    def _has_video(post_with_context: PostWithContext) -> bool:
        for post in [post_with_context.post, post_with_context.quoted_post, post_with_context.in_reply_to_post]:
            if post is None:
                continue
            for media in post.media:
                if media.media_type not in ("photo",):
                    return True
        return False

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        return re.findall(r"https?://[^\s)]+", text)

    @staticmethod
    def _count_note_chars(text: str) -> int:
        """URLを1文字として換算した文字数を返す。"""
        collapsed = re.sub(r"https?://\S+", "X", text)
        return len(collapsed)

    # ── MCP / CLI helpers ──

    def _get_mcp_config_path(self) -> str | None:
        path = Path(self.config.claude_mcp_config_path)
        if path.exists():
            return str(path.resolve())
        return None

    def _build_cli_tool_args(self, allow_web_tools: bool) -> list[str]:
        args: list[str] = []
        mcp_config = self._get_mcp_config_path()
        if mcp_config:
            args += ["--mcp-config", mcp_config]
        if allow_web_tools:
            tools = ["WebSearch", "WebFetch"]
            if mcp_config:
                tools.append("mcp__pdf-reader__*")
            args += ["--allowedTools", ",".join(tools)]
        return args

    # ── LLM call helpers ──

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
                if "model" in option_sig.parameters and self.config.ai_model:
                    option_kwargs["model"] = self.config.ai_model
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

    def _parse_stream_json_output(self, raw_output: str) -> str:
        """Extract the final assistant message from stream-json output."""
        last_assistant_text: str = ""

        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj_type = obj.get("type", "")

            if obj_type == "result":
                texts = self._extract_text_recursive(obj.get("result", {}))
                merged = "\n".join(t for t in texts if t).strip()
                if merged:
                    return merged

            if obj_type == "assistant":
                message = obj.get("message", {})
                content = message.get("content", [])
                texts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "").strip()
                        if t:
                            texts.append(t)
                merged = "\n".join(texts).strip()
                if merged:
                    last_assistant_text = merged

        if last_assistant_text:
            return last_assistant_text

        return raw_output.strip()

    def _run_claude_cli_with_images(
        self, prompt: str, system_prompt: str, images: list[str], *, allow_web_tools: bool = False
    ) -> str:
        content_blocks: list[dict[str, Any]] = []
        for url in images:
            content_blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        full_text = f"{system_prompt}\n\n{prompt}".strip()
        content_blocks.append({"type": "text", "text": full_text})

        stream_input = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content_blocks}},
            ensure_ascii=False,
        )

        cmd = [
            self.config.claude_cli_path, "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if self.config.ai_model:
            cmd += ["--model", self.config.ai_model]
        if self.config.ai_model:
            cmd += ["--model", self.config.ai_model]
        cmd += self._build_cli_tool_args(allow_web_tools)

        proc = subprocess.run(
            cmd, input=stream_input, capture_output=True, text=True,
            timeout=self.config.ai_timeout_sec, check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            raise RuntimeError(
                f"Claude CLI request failed (exit={proc.returncode}): {stderr or 'unknown error'}"
            )

        raw = (proc.stdout or "").strip()
        if not raw:
            raise RuntimeError("Claude CLI returned empty response")
        return self._parse_stream_json_output(raw)

    def _run_claude_cli_prompt(
        self, prompt: str, system_prompt: str, *, allow_web_tools: bool = False, images: list[str] | None = None
    ) -> str:
        if images:
            return self._run_claude_cli_with_images(
                prompt, system_prompt, images, allow_web_tools=allow_web_tools
            )

        full_prompt = f"{system_prompt}\n\n{prompt}".strip()
        cmd = [self.config.claude_cli_path, "--print"]
        if self.config.ai_model:
            cmd += ["--model", self.config.ai_model]
        cmd += self._build_cli_tool_args(allow_web_tools)
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
        self, prompt: str, system_prompt: str, *,
        allow_web_tools: bool = False, images: list[str] | None = None
    ) -> str:
        common = _PROMPTS.get("common_claude_system", {}).get(self.lang, "")
        if common:
            system_prompt = f"{common}\n\n{system_prompt}"
        if images:
            return self._run_claude_cli_prompt(
                prompt, system_prompt, allow_web_tools=allow_web_tools, images=images
            )

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

    # ── Prompts ──

    def _get_prompt_for_pre_filter(self, post_with_context_description: str) -> str:
        return self._p("prompts", "pre_filter").format(
            post_description=post_with_context_description,
        ).strip()

    def _get_prompt_for_live_search(self, post_with_context_description: str) -> str:
        return self._p("prompts", "live_search").format(
            post_description=post_with_context_description,
        ).strip()

    def _get_note_text_rules(self, for_writing: bool = False) -> str:
        rules = self._p("prompts", "note_text_rules")
        if for_writing:
            if self.feed_lang == "all":
                rules += "\n" + self._p("dynamic_rules", "write_in_post_language")
            elif self.lang != "ja":
                rules += "\n" + self._p("dynamic_rules", "write_in_english")
        return rules

    def _get_prompt_for_note_writing(
        self,
        post_with_context_description: str,
        search_results: str,
    ) -> str:
        return self._p("prompts", "note_writing").format(
            note_text_rules=self._get_note_text_rules(for_writing=True),
            post_description=post_with_context_description,
            search_results=search_results,
        ).strip()

    def _get_prompt_for_self_evaluation(
        self,
        post_with_context_description: str,
        note_text: str,
    ) -> str:
        return self._p("prompts", "self_evaluation").format(
            post_description=post_with_context_description,
            note_text=note_text,
        ).strip()

    def _get_prompt_for_note_rewrite(
        self,
        post_with_context_description: str,
        rewrite_history: str,
        search_results: str,
    ) -> str:
        return self._p("prompts", "note_rewrite").format(
            note_text_rules=self._get_note_text_rules(for_writing=True),
            post_description=post_with_context_description,
            rewrite_history=rewrite_history,
            search_results=search_results,
        ).strip()

    def _get_prompt_for_char_reduction_check(
        self,
        post_with_context_description: str,
        original_note: str,
        shortened_note: str,
    ) -> str:
        return self._p("prompts", "char_reduction_check").format(
            post_description=post_with_context_description,
            original_note=original_note,
            shortened_note=shortened_note,
        ).strip()

    def _build_rewrite_history_text(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        parts: list[str] = []
        header = "[リライト履歴]" if self.lang == "ja" else "[Rewrite History]"
        parts.append(header)
        for i, entry in enumerate(history, 1):
            is_latest = i == len(history)
            if self.lang == "ja":
                label = f"--- Round {i} (最新・改善対象) ---" if is_latest else f"--- Round {i} ---"
                parts.append(label)
                parts.append(f"ノート:\n{entry['note']}")
                parts.append(f"\nフィードバック:\n{entry['feedback']}")
            else:
                label = f"--- Round {i} (Latest. Target for improvement) ---" if is_latest else f"--- Round {i} ---"
                parts.append(label)
                parts.append(f"Note:\n{entry['note']}")
                parts.append(f"\nFeedback:\n{entry['feedback']}")
            parts.append("")
        return "\n".join(parts)

    # ── Phase methods ──

    def _pre_filter_post(self, post_with_context: PostWithContext) -> bool:
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)
        prompt = self._get_prompt_for_pre_filter(description)

        try:
            sys_prompt = self._p("system_prompts", "pre_filter")
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    images=image_urls or None,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Pre-filter failed: %s", ex)
            return True

        logger.info("Pre-filter response: %s", raw)
        return "YES" in raw.upper()

    def _run_live_search(
        self, post_with_context_description: str, image_urls: list[str] | None = None
    ) -> str:
        user_prompt = self._get_prompt_for_live_search(post_with_context_description)
        provider = self.config.ai_provider.lower()

        if provider in {"claude", "claude_agent", "claude-agent"}:
            try:
                return self._claude_completion(
                    prompt=user_prompt,
                    system_prompt=self._p("system_prompts", "live_search_claude"),
                    allow_web_tools=True,
                    images=image_urls or None,
                )
            except Exception as ex:
                logger.warning("Live search failed (claude): %s", ex)
                return ""

        sys_prompt_other = self._p("system_prompts", "live_search_other")
        if provider == "xai":
            chat_payload: dict[str, Any] = {
                "model": self.config.ai_model,
                "temperature": 0.6,
                "messages": [
                    {
                        "role": "system",
                        "content": sys_prompt_other,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
            }
            try:
                return self._chat_completion(chat_payload)
            except Exception as ex:
                logger.warning("Live search failed (xai chat): %s", ex)
                responses_payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "input": user_prompt,
                    "tools": [{"type": "web_search"}],
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
                    "content": sys_prompt_other,
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

    def _parse_eval_result(self, raw: str) -> tuple[str, str]:
        """Parse self-evaluation response. Returns (verdict, detail).
        verdict is one of: "pass", "improve", "fail"
        """
        cleaned = raw.strip()

        no_stars = cleaned.replace("*", "")
        verdict_match = re.search(
            r"Verdict\s*:\s*(PASS|IMPROVE|FAIL)",
            no_stars,
            re.IGNORECASE,
        )
        if verdict_match:
            v = verdict_match.group(1).upper()
            if v == "PASS":
                return ("pass", "")
            detail = cleaned[verdict_match.end():]
            detail = detail.lstrip(":").lstrip("*").strip()
            return (v.lower(), detail or cleaned)

        for line in cleaned.split("\n"):
            line = line.strip().strip("*").strip("-").strip()
            if not line:
                continue
            upper = line.upper()

            if upper.startswith("PASS"):
                return ("pass", "")
            if upper.startswith("IMPROVE"):
                detail = line.split(":", 1)[1].strip() if ":" in line else cleaned
                return ("improve", detail)
            if upper.startswith("FAIL"):
                detail = line.split(":", 1)[1].strip() if ":" in line else cleaned
                return ("fail", detail)

        upper_all = cleaned.upper()
        if "FAIL" in upper_all:
            return ("fail", cleaned)
        if "IMPROVE" in upper_all:
            return ("improve", cleaned)
        if "PASS" in upper_all:
            return ("pass", "")
        return ("fail", cleaned)

    def _parse_pass_fail_result(self, raw: str) -> tuple[str, str]:
        """Parse a PASS/FAIL response. Returns (verdict, detail).
        verdict is one of: "pass", "fail"
        """
        cleaned = raw.strip()
        no_stars = cleaned.replace("*", "")

        verdict_match = re.search(
            r"Verdict\s*:\s*(PASS|FAIL)",
            no_stars,
            re.IGNORECASE,
        )
        if verdict_match:
            v = verdict_match.group(1).upper()
            if v == "PASS":
                return ("pass", "")
            detail = cleaned[verdict_match.end():]
            detail = detail.lstrip(":").lstrip("*").strip()
            return ("fail", detail or cleaned)

        for line in cleaned.split("\n"):
            line = line.strip().strip("*").strip("-").strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("PASS"):
                return ("pass", "")
            if upper.startswith("FAIL"):
                detail = line.split(":", 1)[1].strip() if ":" in line else cleaned
                return ("fail", detail)

        upper_all = cleaned.upper()
        if "FAIL" in upper_all:
            return ("fail", cleaned)
        if "PASS" in upper_all:
            return ("pass", "")
        return ("fail", cleaned)

    def _self_evaluate_note(
        self, post_with_context: PostWithContext, note_text: str
    ) -> tuple[str, str, str]:
        """Returns (verdict, detail, raw). verdict is "pass", "improve", or "fail"."""
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)
        prompt = self._get_prompt_for_self_evaluation(description, note_text)

        try:
            sys_prompt = self._p("system_prompts", "self_evaluation")
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    images=image_urls or None,
                    allow_web_tools=True,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Self-evaluation failed: %s", ex)
            return ("pass", "", "")

        logger.info("Self-evaluation response: %s", raw)
        verdict, detail = self._parse_eval_result(raw)
        return (verdict, detail, raw)

    def _check_char_reduction_quality(
        self, post_with_context_description: str, original_note: str, shortened_note: str
    ) -> tuple[str, str]:
        """Returns (verdict, feedback). verdict is "pass" or "fail"."""
        provider = self.config.ai_provider.lower()
        prompt = self._get_prompt_for_char_reduction_check(
            post_with_context_description, original_note, shortened_note
        )
        sys_prompt = self._p("system_prompts", "char_reduction_check")

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(prompt=prompt, system_prompt=sys_prompt)
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Char reduction quality check failed: %s", ex)
            return ("pass", "")

        logger.info("Char reduction check response: %s", raw)
        verdict, detail = self._parse_pass_fail_result(raw)
        return (verdict, detail)

    def _rewrite_note(
        self,
        post_with_context: PostWithContext,
        rewrite_history: list[dict[str, str]],
        search_results: str,
    ) -> str | None:
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)
        history_text = self._build_rewrite_history_text(rewrite_history)
        prompt = self._get_prompt_for_note_rewrite(description, history_text, search_results)

        try:
            sys_prompt = self._p("system_prompts", "note_rewrite")
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    allow_web_tools=True,
                    images=image_urls or None,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Note rewrite failed: %s", ex)
            return None

        logger.info("Rewritten note:\n%s", raw)

        if NO_NOTE_NEEDED in raw.upper():
            return None
        if NOT_ENOUGH_EVIDENCE in raw.upper():
            return None
        if not self._extract_urls(raw):
            return None
        if "#" in raw:
            return None

        return raw.strip()

    # ── Main entry point ──

    def generate_note(self, post_with_context: PostWithContext) -> NoteGenerationResult:
        provider = self.config.ai_provider.lower()
        if provider in {"none", "off", "disabled"}:
            return NoteGenerationResult(draft=None, reason="disabled")

        if self._has_video(post_with_context):
            return NoteGenerationResult(draft=None, reason="has_video")

        if not self._pre_filter_post(post_with_context):
            return NoteGenerationResult(draft=None, reason="no_factual_claims")

        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)

        search_results = self._run_live_search(description, image_urls=image_urls)
        logger.info("Live search results:\n%s", search_results if search_results else "(empty)")

        note_prompt = self._get_prompt_for_note_writing(description, search_results)

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=note_prompt,
                    system_prompt=self._p("system_prompts", "note_writing"),
                    allow_web_tools=True,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.3,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._p("system_prompts", "note_writing_other"),
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

        # ── Self-evaluation with up to 3 rewrites ──
        max_rewrites = 3
        rewrite_count = 0
        rewrite_history: list[dict[str, str]] = []
        verdict, detail, eval_raw = self._self_evaluate_note(post_with_context, note_text)

        while verdict == "improve" and rewrite_count < max_rewrites:
            rewrite_count += 1
            rewrite_history.append({"note": note_text, "feedback": eval_raw})
            logger.info("Self-evaluation requested improvement (rewrite %d/%d): %s", rewrite_count, max_rewrites, detail)
            rewritten = self._rewrite_note(post_with_context, rewrite_history, search_results)
            if rewritten is None:
                draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
                return NoteGenerationResult(draft=draft, reason="rewrite_gave_up", rewrite_count=rewrite_count)

            note_text = rewritten
            verdict, detail, eval_raw = self._self_evaluate_note(post_with_context, note_text)

        if verdict == "improve":
            logger.info("Still IMPROVE after %d rewrites, not submitting", max_rewrites)
            draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
            return NoteGenerationResult(draft=draft, reason="self_eval_improve_after_max_rewrites", rewrite_count=rewrite_count)

        if verdict == "fail":
            draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
            reason = f"self_eval_failed_after_rewrite: {detail}" if rewrite_count > 0 else f"self_eval_failed: {detail}"
            return NoteGenerationResult(draft=draft, reason=reason, rewrite_count=rewrite_count)

        # ── Char count enforcement (post-PASS) ──
        max_char_rewrites = 2
        char_rewrite_count = 0
        pre_shorten_note = note_text

        while self._count_note_chars(note_text) > 280 and char_rewrite_count < max_char_rewrites:
            char_rewrite_count += 1
            char_count = self._count_note_chars(note_text)
            feedback = self._p("prompts", "char_count_feedback").format(char_count=char_count)
            rewrite_history.append({"note": note_text, "feedback": feedback})
            logger.info(
                "Note over char limit (%d/280), rewriting (%d/%d)",
                char_count, char_rewrite_count, max_char_rewrites,
            )
            rewritten = self._rewrite_note(post_with_context, rewrite_history, search_results)
            if rewritten is None:
                break
            note_text = rewritten

        rewrite_count += char_rewrite_count

        if self._count_note_chars(note_text) > 280:
            draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
            return NoteGenerationResult(draft=draft, reason="over_char_limit", rewrite_count=rewrite_count)

        # ── Info loss check after shortening ──
        if pre_shorten_note != note_text:
            quality_verdict, quality_feedback = self._check_char_reduction_quality(
                description, pre_shorten_note, note_text
            )
            if quality_verdict == "fail":
                rewrite_count += 1
                rewrite_history.append({"note": note_text, "feedback": quality_feedback})
                logger.info("Char reduction info loss detected, rewriting: %s", quality_feedback)
                rewritten = self._rewrite_note(post_with_context, rewrite_history, search_results)
                if (
                    rewritten is not None
                    and self._count_note_chars(rewritten) <= 280
                    and self._extract_urls(rewritten)
                    and "#" not in rewritten
                ):
                    note_text = rewritten
                else:
                    draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
                    return NoteGenerationResult(
                        draft=draft, reason="char_reduction_info_loss", rewrite_count=rewrite_count
                    )

        draft = AINoteDraft(
            note_text=note_text,
            misleading_tags=["missing_important_context"],
        )
        return NoteGenerationResult(draft=draft, reason="ok", rewrite_count=rewrite_count)