from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging

from .config import AppConfig
from .service import CommunityNoteWriterService, save_recent_notes, save_summary


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="X Community Notes AI Writer")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Fetch eligible posts and draft/submit notes")
    run_p.add_argument("--num-posts", type=int, default=None)
    run_p.add_argument("--test-mode", type=_parse_bool, default=None)
    run_p.add_argument("--submit-notes", type=_parse_bool, default=None)
    run_p.add_argument(
        "--evaluate-before-submit",
        type=_parse_bool,
        default=None,
    )
    run_p.add_argument("--min-claim-opinion-score", type=float, default=None)
    run_p.add_argument("--enable-url-check", type=_parse_bool, default=None)
    run_p.add_argument("--url-check-timeout", type=int, default=None)
    run_p.add_argument("--feed-lang", type=str, default=None, choices=["ja", "all"])

    notes_p = sub.add_parser("notes", help="Fetch notes written by this account")
    notes_p.add_argument("--test-mode", type=_parse_bool, default=None)
    notes_p.add_argument("--max-results", type=int, default=20)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = AppConfig.from_env()
    logging.basicConfig(level=logging.INFO)
    service = CommunityNoteWriterService(config)

    if args.command == "run":
        summary = service.run_once(
            num_posts=args.num_posts if args.num_posts is not None else config.default_num_posts,
            test_mode=args.test_mode if args.test_mode is not None else config.default_test_mode,
            submit_notes=args.submit_notes if args.submit_notes is not None else config.default_submit_notes,
            evaluate_before_submit=(
                args.evaluate_before_submit
                if args.evaluate_before_submit is not None
                else config.default_evaluate_before_submit
            ),
            min_claim_opinion_score=(
                args.min_claim_opinion_score
                if args.min_claim_opinion_score is not None
                else config.default_min_claim_opinion_score
            ),
            enable_url_check=(
                args.enable_url_check
                if args.enable_url_check is not None
                else config.default_enable_url_check
            ),
            url_check_timeout_sec=(
                args.url_check_timeout
                if args.url_check_timeout is not None
                else config.url_check_timeout_sec
            ),
            feed_lang=args.feed_lang or "ja",
        )
        path = save_summary(summary)
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        print(f"Saved: {path}")
        return

    if args.command == "notes":
        notes = service.fetch_recent_notes(max_results=args.max_results, test_mode=args.test_mode if args.test_mode is not None else config.default_test_mode)
        path = save_recent_notes(notes)
        print(json.dumps(notes, ensure_ascii=False, indent=2))
        print(f"Saved: {path}")
        return


if __name__ == "__main__":
    main()