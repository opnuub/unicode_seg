#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

# Myanmar (Burmese) Unicode block: U+1000–U+109F
BURMESE_BLOCK = r"\u1000-\u109F"

def build_pattern(only_burmese_and_pipe: bool, keep_newlines: bool) -> re.Pattern:
    """
    Returns a regex that matches characters to REMOVE.
    """
    if only_burmese_and_pipe:
        # Keep only Burmese + '|' (+ optionally newlines)
        if keep_newlines:
            return re.compile(rf"[^{BURMESE_BLOCK}\|\n\r]")
        return re.compile(rf"[^{BURMESE_BLOCK}\|]")
    else:
        # Default: keep Burmese + '|' + ALL whitespace (spaces/tabs/newlines)
        return re.compile(rf"[^{BURMESE_BLOCK}\|\s]")

def clean_text(text: str, only_burmese_and_pipe: bool, keep_newlines: bool) -> str:
    pattern = build_pattern(only_burmese_and_pipe, keep_newlines)
    return pattern.sub("", text)

def default_output_path(input_path: Path) -> Path:
    # e.g., "myfile.txt" -> "myfile.cleaned.txt"
    suffix = input_path.suffix or ".txt"
    return input_path.with_name(f"{input_path.stem}.cleaned{suffix}")

def main():
    ap = argparse.ArgumentParser(
        description="Remove all non-Burmese characters from a text file "
                    "(U+1000–U+109F), except the '|' breakpoint."
    )
    ap.add_argument("input", help="Path to input text file")
    ap.add_argument("output", nargs="?", help="Path to output file (optional)")
    ap.add_argument("-i", "--in-place", action="store_true",
                    help="Overwrite the input file with cleaned text")
    ap.add_argument("--only-burmese-and-pipe", action="store_true",
                    help="Keep ONLY Burmese (U+1000–U+109F) and '|'. "
                         "By default, whitespace is preserved too.")
    ap.add_argument("--no-keep-newlines", action="store_true",
                    help="(Use with --only-burmese-and-pipe) Also remove newlines.")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        ap.error(f"Input file not found: {input_path}")

    if args.in_place and args.output:
        ap.error("Use either --in-place or an explicit output path, not both.")

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(
        text,
        only_burmese_and_pipe=args.only_burmese_and_pipe,
        keep_newlines=not args.no_keep_newlines
    )

    if args.in_place:
        input_path.write_text(cleaned, encoding="utf-8")
        print(f"Cleaned in place: {input_path}")
    else:
        out_path = Path(args.output) if args.output else default_output_path(input_path)
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"Wrote cleaned text to: {out_path}")

if __name__ == "__main__":
    main()
