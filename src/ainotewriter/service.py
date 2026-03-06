from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List

from .ai_writer import AINoteGenerator
from .config import AppConfig
from .models import NoteProcessResult, ProposedNote, RunSummary
from .x_client import XCommunityNotesClient
import requests
import html
import re
import logging

logger = logging.getLogger(__name__)


def _send_discord_notification(
    webhook_url: str,
    post_id: str,
    note_text: str,
    evaluation: dict[str, Any] | None,
    test_mode: bool,
    progress_label: str,
) -> None:
    if not webhook_url:
        return

    embeds: list[dict[str, Any]] = [
        {
            "title": "ノート案",
            "description": note_text[:4096],
            "color": 0x1DA1F2,
        }
    ]

    if evaluation is not None:
        data = evaluation.get("data", {}) if isinstance(evaluation, dict) else {}
        score = data.get("claim_opinion_score")
        lines: list[str] = []
        if score is not None:
            lines.append(f"claim_opinion_score: **{score}**")
        for key, value in data.items():
            if key == "claim_opinion_score":
                continue
            lines.append(f"{key}: {value}")
        if not lines:
            lines.append("(データなし)")
        embeds.append(
            {
                "title": "評価結果",
                "description": "\n".join(lines)[:4096],
                "color": 0x2ECC71 if score is not None and score < 0.55 else 0xE74C3C,
            }
        )

    mode_label = "テストモード" if test_mode else "本番モード"
    embeds[-1]["footer"] = {"text": f"{mode_label} | {progress_label}"}

    payload = {
        "content": f"https://x.com/i/status/{post_id}",
        "embeds": embeds,
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if not resp.ok:
            logger.warning("Discord webhook failed (%s): %s", resp.status_code, resp.text)
    except Exception as ex:
        logger.warning("Discord webhook error: %s", ex)


class CommunityNoteWriterService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.x_client = XCommunityNotesClient(config)
        self.ai = AINoteGenerator(config)
        self.fixed_post_selection = "feed_lang:ja"

    def run_once(
        self,
        num_posts: int,
        test_mode: bool,
        submit_notes: bool,
        evaluate_before_submit: bool,
        min_claim_opinion_score: float,
        enable_url_check: bool = False,
        url_check_timeout_sec: int = 5,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RunSummary:
        def _progress(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)
            else:
                logging.getLogger(__name__).info(message)

        def _extract_post_id_from_written_note(item: dict[str, Any]) -> str | None:
            info = item.get("info") if isinstance(item, dict) else None
            if isinstance(info, dict):
                post_id = info.get("post_id")
                if isinstance(post_id, str) and post_id:
                    return post_id

            for key in ("post_id", "tweet_id", "target_post_id"):
                value = item.get(key) if isinstance(item, dict) else None
                if isinstance(value, str) and value:
                    return value
            return None

        started_at = datetime.utcnow().isoformat() + "Z"
        _progress("Run started")
        _progress("Fetching posts eligible for notes...")
        posts = self.x_client.get_posts_eligible_for_notes(
            max_results=num_posts,
            test_mode=test_mode,
            post_selection=self.fixed_post_selection,
        )
        _progress(f"Fetched {len(posts)} posts")

        _progress("Fetching already written notes...")
        written_notes = self.x_client.get_notes_written(max_results=100, test_mode=test_mode)
        written_data = written_notes.get("data", []) if isinstance(written_notes, dict) else []
        already_noted_post_ids = {
            post_id
            for post_id in (_extract_post_id_from_written_note(item) for item in written_data if isinstance(item, dict))
            if post_id
        }
        _progress(f"Found {len(already_noted_post_ids)} posts already noted")

        results: list[NoteProcessResult] = []
        processed_post_ids: set[str] = set()
        for idx, pwc in enumerate(posts, start=1):
            try:
                _progress(f"[{idx}/{len(posts)}] Processing post_id={pwc.post.post_id}")
                _progress(f"Original post: {pwc.post.text}")

                if pwc.post.post_id in processed_post_ids:
                    _progress("Skipped: duplicate in current run")
                    continue
                processed_post_ids.add(pwc.post.post_id)

                if pwc.post.post_id in already_noted_post_ids:
                    _progress("Skipped: note already submitted for this post")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="skipped",
                            reason="Already submitted note for this post",
                        )
                    )
                    continue

                _progress("Generating note draft...")
                gen_result = self.ai.generate_note(pwc)
                draft = gen_result.draft
                reason = gen_result.reason

                is_self_eval_skip = reason.startswith("self_eval_failed") or reason == "rewrite_gave_up"

                if draft is None or is_self_eval_skip:
                    _progress(f"Skipped: {reason}")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="skipped",
                            reason=reason,
                            generated_note=draft.note_text if draft else None,
                        )
                    )
                    continue

                score = None
                evaluation = None
                if evaluate_before_submit:
                    _progress("Evaluating note quality...")
                    try:
                        evaluation = self.x_client.evaluate_note(
                            post_id=pwc.post.post_id,
                            note_text=draft.note_text,
                        )
                        logger.info("Evaluation response: %s", evaluation)
                        score = (
                            evaluation.get("data", {}).get("claim_opinion_score")
                            if isinstance(evaluation, dict)
                            else None
                        )
                        _progress(f"claim_opinion_score={score}")
                    except Exception as eval_ex:
                        _progress(f"Evaluation failed: {eval_ex}")
                        evaluation = {"error": str(eval_ex)}

                progress_label = f"{idx}/{len(posts)}"
                _send_discord_notification(
                    webhook_url=self.config.discord_webhook_url,
                    post_id=pwc.post.post_id,
                    note_text=draft.note_text,
                    evaluation=evaluation,
                    test_mode=test_mode,
                    progress_label=progress_label,
                )

                if score is not None and score < min_claim_opinion_score:
                    _progress(f"Skipped: claim_opinion_score too low ({score})")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="skipped",
                            reason=f"claim_opinion_score too low: {score}",
                            generated_note=draft.note_text,
                            claim_opinion_score=score,
                        )
                    )
                    continue

                if submit_notes:
                    _progress("Submitting note...")

                    if enable_url_check:
                        parsed_variants = _extract_urls(unescape(draft.note_text))
                        ok = True
                        bad_urls: list[str] = []
                        if parsed_variants:
                            def _check_fn(url: str) -> bool:
                                try:
                                    resp = requests.head(url, allow_redirects=True, timeout=url_check_timeout_sec)
                                    if resp.status_code >= 400:
                                        resp = requests.get(url, allow_redirects=True, timeout=url_check_timeout_sec)
                                    return 200 <= resp.status_code < 400
                                except Exception:
                                    return False

                            ok, bad_urls = check_all_urls_for_note(draft.note_text, _check_fn)
                        else:
                            urls = getattr(pwc.post, "suggested_source_links", []) or []
                            if urls:
                                ok, bad_urls = self._check_urls(urls, url_check_timeout_sec)

                        _progress(f"URL check: ok={ok}, bad={bad_urls}")

                        if not ok:
                            _progress(f"Skipped: invalid URLs: {', '.join(bad_urls)}")
                            results.append(
                                NoteProcessResult(
                                    post_id=pwc.post.post_id,
                                    status="skipped",
                                    reason=f"Invalid URLs: {', '.join(bad_urls)}",
                                    generated_note=draft.note_text,
                                    claim_opinion_score=score,
                                )
                            )
                            continue

                    note = ProposedNote(
                        post_id=pwc.post.post_id,
                        note_text=draft.note_text,
                        misleading_tags=draft.misleading_tags,
                        trustworthy_sources=True,
                    )
                    submission = self.x_client.submit_note(note=note, test_mode=test_mode)
                    _progress("Submitted")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="submitted",
                            generated_note=draft.note_text,
                            claim_opinion_score=score,
                            submission_response=submission,
                        )
                    )
                else:
                    _progress(f"Draft created (submit_notes=False):\n{draft.note_text}")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="drafted",
                            generated_note=draft.note_text,
                            claim_opinion_score=score,
                        )
                    )
            except Exception as ex:
                _progress(f"Error: {ex}")
                results.append(
                    NoteProcessResult(
                        post_id=pwc.post.post_id,
                        status="error",
                        reason=str(ex),
                    )
                )

        finished_at = datetime.utcnow().isoformat() + "Z"
        _progress("Run finished")
        return RunSummary(
            started_at=started_at,
            finished_at=finished_at,
            test_mode=test_mode,
            submit_notes=submit_notes,
            evaluate_before_submit=evaluate_before_submit,
            num_posts_requested=num_posts,
            num_posts_fetched=len(posts),
            results=results,
        )

    def fetch_recent_notes(self, max_results: int = 20, test_mode: bool = True) -> dict:
        notes = self.x_client.get_notes_written(max_results=max_results, test_mode=test_mode)

        data = notes.get("data", []) if isinstance(notes, dict) else []
        if isinstance(data, list):
            notes["data"] = sorted(data, key=self._note_sort_key)

        stats_source = self.x_client.get_notes_written(max_results=50, test_mode=True)
        recent_data = stats_source.get("data", []) if isinstance(stats_source, dict) else []
        if isinstance(recent_data, list):
            most_recent_50 = sorted(recent_data, key=self._note_sort_key, reverse=True)[:50]
        else:
            most_recent_50 = []

        notes["compliance_stats"] = self._build_compliance_stats(most_recent_50)
        return notes

    @staticmethod
    def _note_sort_key(item: dict[str, Any]) -> tuple[int, str]:
        note_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(note_id, str):
            try:
                return (0, str(int(note_id)))
            except ValueError:
                return (1, note_id)
        return (2, "")

    @staticmethod
    def _extract_bucket(item: dict[str, Any], evaluator_type: str) -> str | None:
        test_result = item.get("test_result") if isinstance(item, dict) else None
        outcomes = test_result.get("evaluation_outcome", []) if isinstance(test_result, dict) else []
        if not isinstance(outcomes, list):
            return None

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            if outcome.get("evaluator_type") != evaluator_type:
                continue
            bucket = outcome.get("evaluator_score_bucket")
            if isinstance(bucket, str):
                return bucket.lower()
        return None

    def _build_compliance_stats(self, notes: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(notes)

        def _count(evaluator: str, bucket: str) -> int:
            return sum(
                1
                for item in notes
                if self._extract_bucket(item, evaluator) == bucket.lower()
            )

        claim_high = _count("ClaimOpinion", "high")
        claim_low = _count("ClaimOpinion", "low")
        url_high = _count("UrlValidity", "high")
        harassment_high = _count("HarassmentAbuse", "high")

        def _rate(count: int) -> float:
            return (count / total * 100.0) if total > 0 else 0.0

        claim_high_rate = _rate(claim_high)
        claim_low_rate = _rate(claim_low)
        url_high_rate = _rate(url_high)
        harassment_high_rate = _rate(harassment_high)

        return {
            "basis": "most_recent_50_test_mode_notes",
            "sample_size": total,
            "metrics": {
                "claim_opinion_high": {
                    "count": claim_high,
                    "rate_percent": round(claim_high_rate, 2),
                    "threshold": "<= 30.0%",
                    "passed": claim_high_rate <= 30.0,
                },
                "claim_opinion_low": {
                    "count": claim_low,
                    "rate_percent": round(claim_low_rate, 2),
                    "threshold": ">= 30.0%",
                    "passed": claim_low_rate >= 30.0,
                },
                "url_validity_high": {
                    "count": url_high,
                    "rate_percent": round(url_high_rate, 2),
                    "threshold": ">= 95.0%",
                    "passed": url_high_rate >= 95.0,
                },
                "harassment_abuse_high": {
                    "count": harassment_high,
                    "rate_percent": round(harassment_high_rate, 2),
                    "threshold": ">= 98.0%",
                    "passed": harassment_high_rate >= 98.0,
                },
            },
            "all_requirements_passed": (
                claim_high_rate <= 30.0
                and claim_low_rate >= 30.0
                and url_high_rate >= 95.0
                and harassment_high_rate >= 98.0
            ),
        }

    def _check_urls(self, urls: list[str], timeout_sec: int) -> tuple[bool, list[str]]:
        bad: list[str] = []
        for url in urls:
            if not isinstance(url, str) or not url:
                bad.append(str(url))
                continue
            try:
                resp = requests.head(url, allow_redirects=True, timeout=timeout_sec)
                if resp.status_code >= 400:
                    resp = requests.get(url, allow_redirects=True, timeout=timeout_sec)
                if not (200 <= resp.status_code < 400):
                    bad.append(f"{url} (status={resp.status_code})")
            except Exception as ex:
                bad.append(f"{url} ({ex.__class__.__name__})")
        return (len(bad) == 0, bad)


def check_all_urls_for_note(note_text: str, check_url_fn: Callable[[str], bool]) -> tuple[bool, list[str]]:
    note_text_unescaped = unescape(note_text)
    urls = _extract_urls(note_text_unescaped)

    if len(urls) == 0:
        logger.info(f"No URLs found in note text: {note_text_unescaped}")
        return (False, [])

    bad: list[str] = []
    for url_variant_list in urls:
        at_least_one_good_url_variant = False
        for url_variant in url_variant_list:
            if check_url_fn(url_variant):
                at_least_one_good_url_variant = True
                break
        if not at_least_one_good_url_variant:
            bad.append(" / ".join(url_variant_list))
    if bad:
        logger.info(
            f"No valid URL found for any variant of {bad} in note text: {note_text_unescaped}"
        )
        return (False, bad)
    logger.info(f"All URLs in note text are valid: {note_text_unescaped}")
    return (True, [])


def _extract_urls(text: str) -> List[List[str]]:
    pattern = re.compile(
        r"""
            (?:
                https?://               # optional scheme
            )?
            (?:www\.)?                  # optional www
            [\w\-._~%]+                 # subdomain or domain name chars
            \.[a-zA-Z]{2,}              # dot + top level domain
            (?:/[^\s]*)?                # optional query/fragment
            """,
        re.VERBOSE,
    )

    raw_matches = pattern.findall(text)

    strip_trailing = ".,;:!?)]}\"'"
    results = []
    for match in raw_matches:
        variants = [match]
        stripped_variant = match.rstrip(strip_trailing)
        if stripped_variant != match:
            variants.append(stripped_variant)
        results.append(variants)
    return results


def unescape(text: str) -> str:
    return html.unescape(html.unescape(text)) if isinstance(text, str) else text


def save_summary(summary: RunSummary, output_dir: str = "outputs") -> Path:
    run_dir = Path(output_dir) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = run_dir / f"run_{timestamp}.json"
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_recent_notes(notes: dict, output_dir: str = "outputs") -> Path:
    notes_dir = Path(output_dir) / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = notes_dir / f"notes_{timestamp}.json"
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return path