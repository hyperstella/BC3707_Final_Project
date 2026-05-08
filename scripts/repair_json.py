"""
JSON repair pipeline for malformed dialogue files.

Handles every corruption pattern found in this dataset:
  1. Trailing commas before } or ]
  2. Invalid backslash escapes  (LaTeX math: \\{, \\}, \\sigma, \\frac, ...)
  3. Unicode curly/smart quotes used as JSON delimiters
  4. Missing commas between adjacent object properties
  5. Multi-line string content  (literal newline inside a JSON string)
  6. Stray bare values inside objects  (orphaned array elements)

Usage:
  python scripts/repair_json.py                    # repair all *.json in dataset dir
  python scripts/repair_json.py path/to/file.json  # repair a single file
  python scripts/repair_json.py --check-only       # report which files are broken
  python scripts/repair_json.py --dry-run          # show diffs without writing

New data: drop JSON files into the dataset directory and re-run this script.
"""

import argparse
import json
import os
import re
import sys
import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Individual repair passes ──────────────────────────────────────────────────

def pass_curly_quotes(text: str) -> str:
    """Replace Unicode "smart" quote chars with plain ASCII double quotes."""
    for ch in '""„“”„«»':
        text = text.replace(ch, '"')
    for ch in "''‘’":
        text = text.replace(ch, "'")
    return text


def pass_trailing_commas(text: str) -> str:
    """Remove commas immediately before a closing } or ]."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def pass_invalid_escapes(text: str) -> str:
    """
    Fix backslash sequences that are not valid JSON escapes.
    Valid: backslash + one of: double-quote, backslash, slash, b, f, n, r, t, uXXXX
    Everything else: double the backslash so the sequence becomes a literal backslash.
    """
    VALID_NEXT = set('"\\' + '/bfnrtu')
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == 'u' and i + 5 < len(text):
                # \uXXXX — copy all 6 chars and skip past them
                result.append(text[i: i + 6])
                i += 6
            elif nxt in VALID_NEXT:
                # Valid two-char escape — copy both and advance past both
                result.append(text[i])
                result.append(nxt)
                i += 2
            else:
                # Invalid escape — double the leading backslash only
                result.append('\\\\')
                i += 1          # leave nxt to be processed normally next iteration
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def pass_missing_commas(text: str) -> str:
    """
    Insert a comma when a closing string/number/bool is immediately followed
    by a newline and then a new property key "word": on the next line.
    Handles cases like:
        "notes": ""
        "score": 30
    """
    # After a value (string end, number, true/false/null) followed by newline + whitespace + "key":
    text = re.sub(
        r'(\"[^\"]*\"|[0-9]+|true|false|null)(\s*\n(\s+)\")',
        lambda m: m.group(1) + ',' + m.group(2),
        text,
    )
    return text


def pass_multiline_strings(text: str) -> str:
    """
    Join string values that were split across physical lines with a literal newline.
    A JSON string may not contain a bare newline; it must be \\n.

    Strategy: scan for an opening quote that starts a value (after : or [,),
    and if the string doesn't close on the same line, fold continuation lines
    into it by replacing the bare newline with \\n.
    """
    lines = text.split('\n')
    result = []
    in_string = False
    buf = []

    for line in lines:
        if not in_string:
            # Count unescaped quotes to detect if a string is opened and not closed
            stripped = line
            # Quick heuristic: if line contains a value-starting quote that isn't closed
            open_count = 0
            i = 0
            while i < len(stripped):
                if stripped[i] == '\\':
                    i += 2
                    continue
                if stripped[i] == '"':
                    open_count += 1
                i += 1

            if open_count % 2 == 1:
                # Odd number of quotes → a string was opened but not closed
                in_string = True
                buf = [line]
            else:
                result.append(line)
        else:
            # We're inside an unclosed string — fold this line in
            buf.append(line)
            # Check if the string closed on this line
            open_count = 0
            i = 0
            while i < len(line):
                if line[i] == '\\':
                    i += 2
                    continue
                if line[i] == '"':
                    open_count += 1
                i += 1
            if open_count % 2 == 1:
                # String closed — join the buffer with \n
                result.append('\\n'.join(buf))
                buf = []
                in_string = False

    if buf:
        result.extend(buf)

    return '\n'.join(result)


def pass_stray_bare_values(text: str) -> str:
    """
    Remove orphaned bare string values inside JSON objects.
    Pattern: a comma + optional whitespace + "bare string" NOT followed by a colon.
    E.g.  "Goal": "A",\n\t"B"\n  →  "Goal": "A"\n
    """
    text = re.sub(r',(\s*"[^"]*")(\s*)(?![\s]*:)', r'\2', text)
    return text


def pass_missing_object_separators(text: str) -> str:
    """Add missing comma between adjacent object/array elements: } { or } [ ."""
    text = re.sub(r'(\})([ \t]*\n[ \t]*\{)', r'\1,\2', text)
    text = re.sub(r'(\])([ \t]*\n[ \t]*\[)', r'\1,\2', text)
    return text


def pass_empty_values(text: str) -> str:
    """Replace bare 'key': <newline> (value missing) with 'key': null."""
    text = re.sub(r'(:\s*)\n(\s*[,}\]])', r'\1null\n\2', text)
    return text


def pass_escape_inner_quotes(text: str) -> str:
    """
    After multiline joining, a string value may contain literal unescaped
    double-quotes, e.g.  "content": "He said "hi" to me"
    Strategy per line: find "key": "value"[,] and escape any unescaped "
    inside the value portion (between the outer delimiters).
    """
    lines = text.splitlines()
    result = []
    for line in lines:
        # Match a full key-value pair where value is a quoted string
        # Use a greedy inner group so we capture up to the LAST closing "
        m = re.match(r'^(\s*"(?:[^"\\]|\\.)*"\s*:\s*)"(.*)"(,?\s*)$', line)
        if m:
            prefix  = m.group(1)   # e.g.   "content":
            content = m.group(2)   # everything between outer " "
            suffix  = m.group(3)   # trailing comma / whitespace
            # Escape any unescaped " inside the content
            fixed = re.sub(r'(?<!\\)"', r'\\"', content)
            if fixed != content:
                line = f'{prefix}"{fixed}"{suffix}'
        result.append(line)
    return '\n'.join(result)


# ── Attempt chain ─────────────────────────────────────────────────────────────

PASSES = [
    ("curly_quotes",             pass_curly_quotes),
    ("stray_values",             pass_stray_bare_values),
    ("multiline_strings",        pass_multiline_strings),
    ("trailing_commas",          pass_trailing_commas),
    ("invalid_escapes",          pass_invalid_escapes),
    ("escape_inner_quotes",      pass_escape_inner_quotes),
    ("missing_object_seps",      pass_missing_object_separators),
    ("empty_values",             pass_empty_values),
    ("missing_commas",           pass_missing_commas),
    # Second round — earlier passes can expose new issues
    ("trailing_commas_2",        pass_trailing_commas),
    ("missing_commas_2",         pass_missing_commas),
    ("missing_object_seps_2",    pass_missing_object_separators),
]


def try_load(text: str):
    for strict in (False, True):
        try:
            return json.loads(text, strict=strict), None
        except json.JSONDecodeError as e:
            last_err = e
    return None, last_err


def repair(text: str) -> tuple[str, bool, str]:
    """
    Apply repair passes in sequence, stopping as soon as the file parses.
    Returns (repaired_text, success, message).
    """
    obj, err = try_load(text)
    if obj is not None:
        return text, True, "already valid"

    working = text
    applied = []
    for name, fn in PASSES:
        working = fn(working)
        applied.append(name)
        obj, err = try_load(working)
        if obj is not None:
            return working, True, f"fixed after: {', '.join(applied)}"

    return working, False, f"still broken after all passes: {err}"


# ── File-level operations ─────────────────────────────────────────────────────

def process_file(path: str, dry_run: bool = False, check_only: bool = False) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        original = f.read()

    obj, err = try_load(original)
    if obj is not None:
        if check_only:
            print(f"  OK       {os.path.basename(path)}")
        return True

    if check_only:
        print(f"  BROKEN   {os.path.basename(path)} — {err.msg}")
        return False

    repaired, success, msg = repair(original)

    if dry_run:
        if success:
            # Show a short diff summary
            orig_lines = original.splitlines()
            rep_lines  = repaired.splitlines()
            changed = sum(1 for a, b in zip(orig_lines, rep_lines) if a != b)
            changed += abs(len(orig_lines) - len(rep_lines))
            print(f"  [dry-run] {os.path.basename(path)}: {msg} ({changed} lines changed)")
        else:
            print(f"  [dry-run] FAILED {os.path.basename(path)}: {msg}")
        return success

    if success:
        with open(path, "w", encoding="utf-8") as f:
            f.write(repaired)
        print(f"  FIXED    {os.path.basename(path)} — {msg}")
    else:
        print(f"  FAILED   {os.path.basename(path)} — {msg}")

    return success


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Repair malformed JSON dialogue files.")
    parser.add_argument("files", nargs="*", help="Specific files to repair (default: all *.json in dataset dir)")
    parser.add_argument("--check-only", action="store_true", help="Report broken files without changing them")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    else:
        paths = sorted(glob.glob(os.path.join(BASE, "Final Dataset", "*.json")))

    total = broken_before = fixed = still_broken = 0
    for path in paths:
        total += 1
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        _, err = try_load(raw)
        if err:
            broken_before += 1
            ok = process_file(path, dry_run=args.dry_run, check_only=args.check_only)
            if ok:
                fixed += 1
            else:
                still_broken += 1
        elif args.check_only:
            print(f"  OK       {os.path.basename(path)}")

    if not args.check_only:
        print(f"\n{total} files scanned, {broken_before} broken, "
              f"{fixed} fixed, {still_broken} still broken.")


if __name__ == "__main__":
    main()
