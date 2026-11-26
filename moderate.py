"""Command-line interface for the moderation engine.

Examples (Windows paths with spaces need quotes)::

    python moderate.py selftest
    python moderate.py check --title "Raba nzuri" --description "Size 42" --file "C:\\Photos\\raba 1.jpg"
    python moderate.py check --title "Video ya bidhaa" --file listing.mp4 --json
    python moderate.py block-image "C:\\Banned\\photo.jpg" --reason DRUGS
    python moderate.py reload-lexicon
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from config.settings import enable_utf8_console

SELFTEST_CASES = [
    ("Unga wa sembe 5kg", "Unga safi wa sembe kwa ugali", "ALLOW"),
    ("Kobe Bryant shoes", "Nike Kobe 6 original", "ALLOW"),
    ("Glue gun 40W", "Kwa kazi za mikono", "ALLOW"),
    ("Nauza b4ng1 kali", "Mzigo umefika", "BLOCK"),
    ("Bastola ya kujilinda", "Na risasi 50", "BLOCK"),
    ("Pembe za ndovu", "Kilo 10", "BLOCK"),
    ("Gauni la kitenge", "Nicheki WhatsApp 0712 345 678", "REVIEW"),
]


def _print_result(result: dict, as_json: bool) -> None:
    """Print a moderation result as a short summary or full JSON."""
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    print(f"Decision : {result['decision']}  (score {result['score']:.2f}, {result['processing_ms']} ms)")
    if result["categories"]:
        print(f"Category : {', '.join(result['categories'])}")
    for reason in result["reasons"]:
        where = f" [{reason['file']}]" if reason.get("file") else ""
        print(f"  - {reason['level']:<8} {reason['category']:<22} {reason['detail']}{where}")
    print(f"Muuzaji  : {result['seller_message_sw']}")
    off = {k: v for k, v in result["checks"].items() if v not in ("ready", "not_loaded")}
    if off:
        print(f"Checks   : {off}")


def cmd_check(args: argparse.Namespace) -> int:
    """Moderate one listing given on the command line."""
    from services.moderation_layer1 import ContentModerationLayer1, MediaFile

    files: List[MediaFile] = []
    for name in args.file or []:
        path = Path(name)
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 2
        files.append(MediaFile(path.name, path.read_bytes()))
    result = ContentModerationLayer1().moderate(args.title, args.description or "", files)
    _print_result(result, args.json)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run built-in sample listings and report model status."""
    from services.model_registry import model_status
    from services.moderation_layer1 import ContentModerationLayer1

    layer1 = ContentModerationLayer1()
    failures = 0
    started = time.perf_counter()
    for title, description, expected in SELFTEST_CASES:
        decision = layer1.moderate(title, description)["decision"]
        ok = decision == expected
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {decision:<8} (expected {expected:<6}) {title}")
    print(f"\n{len(SELFTEST_CASES) - failures}/{len(SELFTEST_CASES)} passed in {time.perf_counter() - started:.1f}s")
    print("Model status:")
    for key, info in model_status().items():
        print(f"  {key:<16} {info['status']:<12} {info.get('model', '')} {info.get('error', '')}")
    return 1 if failures else 0


def cmd_block_image(args: argparse.Namespace) -> int:
    """Add an image to the perceptual-hash blocklist."""
    from PIL import Image

    from services.image_analyzer import add_to_blocklist

    phash = add_to_blocklist(Image.open(args.image).convert("RGB"), args.reason)
    print(f"Added {args.image} to the blocklist (pHash {phash}, reason {args.reason})")
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Validate lexicon.json after editing it."""
    from services.lexicon import get_lexicon

    lex = get_lexicon(reload=True)
    print(f"lexicon.json OK: {len(lex.categories)} categories: {', '.join(lex.categories)}")
    return 0


def main(argv: List[str] = None) -> int:
    """CLI entry point."""
    enable_utf8_console()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="show info logs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run sample listings and show model status").set_defaults(func=cmd_selftest)
    check = sub.add_parser("check", help="moderate one listing")
    check.add_argument("--title", required=True)
    check.add_argument("--description", default="")
    check.add_argument("--file", action="append", help="image or video (repeatable)")
    check.add_argument("--json", action="store_true", help="print the full JSON result")
    check.set_defaults(func=cmd_check)
    block = sub.add_parser("block-image", help="add a banned image to the pHash blocklist")
    block.add_argument("image")
    block.add_argument("--reason", default="BANNED_IMAGE")
    block.set_defaults(func=cmd_block_image)
    sub.add_parser("reload-lexicon", help="validate config/lexicon.json").set_defaults(func=cmd_reload)
    args = parser.parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
