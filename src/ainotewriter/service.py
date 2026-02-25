from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ai_writer import AINoteGenerator
from .config import AppConfig
from .models import NoteProcessResult, ProposedNote, RunSummary
from .x_client import XCommunityNotesClient


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
        progress_callback: Callable[[str], None] | None = None,
    ) -> RunSummary:
        def _progress(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)
            else:
                print(message)

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
        for idx, pwc in enumerate(posts, start=1):
            try:
                _progress(f"[{idx}/{len(posts)}] Processing post_id={pwc.post.post_id}")
                _progress(f"Original post: {pwc.post.text}")

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
                draft = self.ai.generate_note(pwc)
                if draft is None:
                    _progress("Skipped: no reliable note generated")
                    results.append(
                        NoteProcessResult(
                            post_id=pwc.post.post_id,
                            status="skipped",
                            reason="No reliable note generated",
                        )
                    )
                    continue

                score = None
                if evaluate_before_submit:
                    _progress("Evaluating note quality...")
                    evaluation = self.x_client.evaluate_note(
                        post_id=pwc.post.post_id,
                        note_text=draft.note_text,
                    )
                    score = (
                        evaluation.get("data", {}).get("claim_opinion_score")
                        if isinstance(evaluation, dict)
                        else None
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
                    _progress("Draft created (submit_notes=False)")
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

        # Compliance check must be based on most recent 50 notes in test_mode.
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
