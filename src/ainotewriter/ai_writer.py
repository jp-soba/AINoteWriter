from __future__ import annotations

import asyncio
import importlib
import inspect
import json
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

    # ── Post description / media helpers ──

    @staticmethod
    def _build_post_description(post_with_context: PostWithContext) -> str:
        lines: list[str] = []
        lines.append("[Target Post]")
        lines.append(post_with_context.post.text)

        if post_with_context.post.media:
            photo_count = sum(1 for m in post_with_context.post.media if m.media_type == "photo")
            video_count = sum(1 for m in post_with_context.post.media if m.media_type != "photo")
            if photo_count:
                lines.append(f"(この投稿には{photo_count}枚の画像が添付されています)")
            if video_count:
                lines.append("(この投稿には動画が含まれていますが、内容は確認できません)")

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
        """Extract only the final assistant message from stream-json output.
        
        Intermediate outputs (tool calls, tool results, agent thinking)
        are filtered out since they are already reflected in the final summary.
        """
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

            # Final result object
            if obj_type == "result":
                texts = self._extract_text_recursive(obj.get("result", {}))
                merged = "\n".join(t for t in texts if t).strip()
                if merged:
                    return merged

            # Assistant message - keep overwriting so we end up with the last one
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

        # Fallback: return raw output if parsing failed
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
        if allow_web_tools:
            cmd += ["--allowedTools", "WebSearch,WebFetch"]

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
        self, prompt: str, system_prompt: str, *,
        allow_web_tools: bool = False, images: list[str] | None = None
    ) -> str:
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
        return f"""この投稿にファクトチェック可能な具体的な事実主張が含まれているかを判定してください。

検証可能な具体的な事実主張（数値、日付、出来事、人物の行動、因果関係など）が含まれており、
かつその主張が誤解を招く/誤りである可能性が少しでもある場合は「YES」と答えてください。
そうでない場合は、「NO」と答えてください。

以下に該当する場合は「検証可能な具体的な事実主張」とはみなされません:
- 個人の意見・感想・感情表現
- 風刺・皮肉・ユーモア
- 一次情報の共有（当事者またはその代理人が自身の体験や調査結果を報告している）
- 未来の予測
- 他者への罵倒・誹謗中傷

注意: 投稿全体のトーンが感情的・論争的であっても、その中に検証可能な具体的事実主張が1つでも含まれていれば「YES」と判定してください。
トーンではなく内容で判断すること。

「YES」か「NO」のみで回答してください。

投稿:
{post_with_context_description}""".strip()

    def _get_prompt_for_live_search(self, post_with_context_description: str) -> str:
        return f"""以下は X の投稿です。投稿内の主張が誤解を招く可能性があるかを、公開情報で調査してください。

    要件:
    - 事実主張ごとに根拠URLを併記する
    - URLは本文にそのまま書く（「出典:」などの飾りは不要）
    - 不確かな情報は推測せず、確認できる情報のみ示す
    - 故人に対する未確認の噂や性的内容に関する主張は調査対象外とし、検索しないこと
    - 裁判で認定された事実や公式報告のみを対象とすること

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
- ノートは短い方が好まれる傾向にある。ただし、重要な情報は省略しないこと
- 少なくとも1つのURLを本文に含める
- URLはそのまま記載（[Source]やカッコで囲む等の装飾禁止）。URL以外のテキストとは、改行等で区切ること
- ハッシュタグ、絵文字、煽り表現は禁止
- 意見ではなく、検証可能な事実と文脈補足を中心に書く
- 冒頭に「Community Note:」等の前置きは付けない

判断方針:
- 迷う場合は書かない（NO NOTE NEEDED か NOT ENOUGH EVIDENCE を返す）
- 未来予測や主観のみの投稿には原則ノートを書かない
- 信頼性の高い一次情報・公的情報を優先する
- ポストの主張を否定または訂正する具体的な根拠がある場合のみノートを書く
- ポストの主張を単に裏付ける・確認するだけのノートは書かない
- 「～は確認できません」「～の根拠は見つかりません」「～は公開情報から確認できていません」のような、自分の検索範囲の限界に基づく記述は絶対に禁止
- - 投稿が既に適切な留保を含んでいる場合、その留保に追加情報を付け足すノートも不要
- 当事者が自身の体験や調査結果を共有している投稿に対して、公開情報で裏付けが取れないことを理由に疑義を呈してはならない
- 「見つけられなかった」ことは「存在しない」ことの根拠にはならない

投稿コンテキスト:
{post_with_context_description}

調査メモ:
```
{search_results}
```
""".strip()

    def _get_prompt_for_self_evaluation(
        self,
        post_with_context_description: str,
        note_text: str,
    ) -> str:
        return f"""あなたはCommunity Notesの品質評価者です。以下のX投稿と、それに対するノート案を評価してください。
調査結果や背景情報は提供しません。ノートの内容だけで判断してください。
ただし、ノートの正確性を検証するために、Web検索やURLの取得は積極的に行ってください。

評価基準:
1. ノートはポストの主張に対して新しい文脈や訂正を追加していますか？ポストの内容を単に確認・裏付けしているだけのノートは不合格です。
2. このノートは、投稿に賛成する人にも反対する人にも等しく有用ですか？一方の政治的立場だけが喜ぶ内容になっていませんか？
3. ノートに意見・推測・主観的な判断は含まれていませんか？
4. ノートに「～は確認できません」「～の根拠は見つかりません」「～は公開情報から確認できていません」のような、検索範囲の限界に基づく疑義が含まれていませんか？
5. ノートの情報は、ポストを見た一般の読者にとって有益な文脈を提供していますか？
6. ノートに含まれるURLを実際に取得し、ノートの記述がソースの内容と一致しているか検証してください。ソースに書かれていない情報がノートに含まれている場合は不合格です。
7. ノートに含まれるすべての事実主張が、ノート内のいずれかの URL で裏付けられている必要があります。URL に記載のない主張がノートに含まれている場合は IMPROVE とし、その主張の根拠 URL の追加を指示してください。
8. ノートに含まれる事実主張を独自に検索し、正確かどうか検証してください。ノートに不正確な主張が含まれている場合は不合格です。

判定:
- すべての基準を満たす場合は「PASS」のみを返してください。
- ノートの方向性は正しいが表現に改善の余地がある場合は「IMPROVE: 具体的な改善指示」を返してください。
- 根本的にノートを書くべきでない場合（投稿を裏付けているだけ、一方的に党派的、等）のみ「FAIL: 理由」を返してください。

[投稿]
{post_with_context_description}

[ノート案]
{note_text}""".strip()

    def _get_prompt_for_note_rewrite(
        self,
        post_with_context_description: str,
        original_note: str,
        feedback: str,
        search_results: str,
    ) -> str:
        return f"""あなたはCommunity Notesの下書きを改善します。以下のフィードバックをもとに、ノートを書き直してください。

書き直しルール:
- フィードバックの指摘を反映する
- それ以外のルールは元のノート作成時と同じ（URLを除いた本文は280文字以内目安、丁寧語、URL必須、意見禁止等）
- 改善できない場合は "NO NOTE NEEDED." と返す
- ノート本文のみを返す（前置き不要）

[投稿]
{post_with_context_description}

[元のノート]
{original_note}

[フィードバック]
{feedback}

[調査メモ]
```
{search_results}
```
""".strip()

    # ── Phase methods ──

    def _pre_filter_post(self, post_with_context: PostWithContext) -> bool:
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)
        prompt = self._get_prompt_for_pre_filter(description)

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt="あなたは投稿の分類器です。指示に従い、YESかNOのみで回答してください。",
                    images=image_urls or None,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "あなたは投稿の分類器です。指示に従い、YESかNOのみで回答してください。"},
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
                    system_prompt="あなたは慎重な調査アシスタントです。公開情報のみを使い、根拠URLを明示してください。",
                    allow_web_tools=True,
                    images=image_urls or None,
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

    def _parse_eval_result(self, raw: str) -> tuple[str, str]:
            """Parse self-evaluation response. Returns (verdict, detail).
            verdict is one of: "pass", "improve", "fail"
            """
            cleaned = raw.strip()

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

            # No explicit verdict found - check entire text
            upper_all = cleaned.upper()
            if "PASS" in upper_all:
                return ("pass", "")
            if "IMPROVE" in upper_all:
                return ("improve", cleaned)
            return ("fail", cleaned)

    def _self_evaluate_note(
        self, post_with_context: PostWithContext, note_text: str
    ) -> tuple[str, str]:
        """Returns (verdict, detail). verdict is "pass", "improve", or "fail"."""
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        image_urls = self._get_image_urls(post_with_context)
        prompt = self._get_prompt_for_self_evaluation(description, note_text)

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt="あなたはCommunity Notesの品質評価者です。厳格に評価してください。",
                    images=image_urls or None,
                    allow_web_tools=True,
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "あなたはCommunity Notesの品質評価者です。厳格に評価してください。"},
                        {"role": "user", "content": prompt},
                    ],
                }
                raw = self._chat_completion(payload)
        except Exception as ex:
            logger.warning("Self-evaluation failed: %s", ex)
            return ("pass", "")

        logger.info("Self-evaluation response: %s", raw)
        return self._parse_eval_result(raw)

    def _rewrite_note(
        self,
        post_with_context: PostWithContext,
        original_note: str,
        feedback: str,
        search_results: str,
    ) -> str | None:
        provider = self.config.ai_provider.lower()
        description = self._build_post_description(post_with_context)
        prompt = self._get_prompt_for_note_rewrite(description, original_note, feedback, search_results)

        try:
            if provider in {"claude", "claude_agent", "claude-agent"}:
                raw = self._claude_completion(
                    prompt=prompt,
                    system_prompt="あなたはCommunity Notes向けの事実確認ライターです。フィードバックを反映してノートを改善してください。",
                )
            else:
                payload: dict[str, Any] = {
                    "model": self.config.ai_model,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": "あなたはCommunity Notes向けの事実確認ライターです。フィードバックを反映してノートを改善してください。"},
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

        # ── Self-evaluation with 1 retry ──
        verdict, detail = self._self_evaluate_note(post_with_context, note_text)

        if verdict == "improve":
            logger.info("Self-evaluation requested improvement: %s", detail)
            rewritten = self._rewrite_note(post_with_context, note_text, detail, search_results)
            if rewritten is None:
                draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
                return NoteGenerationResult(draft=draft, reason="rewrite_gave_up")

            note_text = rewritten
            verdict, detail = self._self_evaluate_note(post_with_context, note_text)

            if verdict == "fail":
                draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
                return NoteGenerationResult(draft=draft, reason=f"self_eval_failed_after_rewrite: {detail}")

        elif verdict == "fail":
            draft = AINoteDraft(note_text=note_text, misleading_tags=["missing_important_context"])
            return NoteGenerationResult(draft=draft, reason=f"self_eval_failed: {detail}")

        draft = AINoteDraft(
            note_text=note_text,
            misleading_tags=["missing_important_context"],
        )
        return NoteGenerationResult(draft=draft, reason="ok")