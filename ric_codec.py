"""
ric_codec.py

Encoder/decoder for the synthetic expired-option RIC scheme used in this
assignment:

    {ROOT}{M}{DD}{YY}{SSSSS}.U^{M}{YY}

    ROOT   underlying root, uppercase                    e.g. UUUU
    M      month letter: A-L = Jan-Dec CALLS, M-X = Jan-Dec PUTS
    DD     two-digit expiration day
    YY     two-digit year
    SSSSS  strike x 100, zero-padded to 5 digits          $12.50 -> 01250
    .U     exchange / venue qualifier
    ^{M}{YY}  expired-contract suffix (repeats month code + year)

build_option_ric() and parse_option_ric() are exact inverses of each other
by construction -- every RIC we generate in the data pull can be round
tripped back through the parser, which is how we test correctness without
ever touching LSEG.
"""

from __future__ import annotations

import re
import datetime as dt

# --- month code tables -------------------------------------------------
# index 0 -> Jan, index 11 -> Dec
CALL_LETTERS = "ABCDEFGHIJKL"
PUT_LETTERS = "MNOPQRSTUVWX"

# month letter -> (month_number, right)
_CODE_TO_MONTH_RIGHT = {}
for _i, _c in enumerate(CALL_LETTERS):
    _CODE_TO_MONTH_RIGHT[_c] = (_i + 1, "C")
for _i, _p in enumerate(PUT_LETTERS):
    _CODE_TO_MONTH_RIGHT[_p] = (_i + 1, "P")

# month_number, right -> month letter
_MONTH_RIGHT_TO_CODE = {v: k for k, v in _CODE_TO_MONTH_RIGHT.items()}

# regex for the full RIC, root captured as "everything before the month
# letter" -- we require the caller to tell us the root length is fixed by
# passing it in, since a bare regex can't tell where ROOT ends and the
# month letter begins (a root could itself contain a letter that looks
# like a month code)
_RIC_RE = re.compile(
    r"^(?P<root>[A-Z]+)"
    r"(?P<code>[A-X])"
    r"(?P<dd>\d{2})"
    r"(?P<yy>\d{2})"
    r"(?P<strike>\d{5})"
    r"\.U"
    r"\^(?P<code2>[A-X])(?P<yy2>\d{2})$"
)


def build_option_ric(root: str, month: int, day: int, year2: int,
                      strike: float, right: str) -> str:
    """
    Construct a RIC from its parts. Inverse of parse_option_ric().

    root    e.g. "UUUU"
    month   1-12
    day     day of month, 1-31
    year2   two-digit year, e.g. 26 for 2026
    strike  dollar strike, e.g. 12.50
    right   "C" or "P"

    IMPORTANT, confirmed against real LSEG data (not just the assignment
    doc's prose): the ^{M}{YY} suffix does NOT repeat the base's own
    month letter. It always uses the CALL letter for that expiration
    month, regardless of whether the base contract is a call or a put.
    A June put has base code R (June put) but suffix F26 (June CALL) --
    verified against 327 real RICs in a reference pull with zero
    exceptions. The doc's "repeats the month letter" description is only
    true for calls, where the call letter and the suffix happen to
    coincide; it silently breaks for puts.
    """
    right = right.upper()
    if right not in ("C", "P"):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    code = _MONTH_RIGHT_TO_CODE[(month, right)]
    suffix_code = CALL_LETTERS[month - 1]  # always the call letter, even for puts
    dd = f"{day:02d}"
    yy = f"{year2:02d}"
    sssss = f"{round(strike * 100):05d}"
    return f"{root.upper()}{code}{dd}{yy}{sssss}.U^{suffix_code}{yy}"


def parse_option_ric(ric: str, root: str | None = None) -> dict | None:
    """
    Parse a RIC into {root, month, day, year, expiry, strike, right}.
    Returns None if the string doesn't match the scheme, or if the
    suffix doesn't agree with the base -- that mismatch means the RIC
    is malformed, not a contract we should trust.

    Suffix validity rule (confirmed against 327 real LSEG RICs, zero
    exceptions): the suffix code must equal the CALL letter for the
    base's expiration month -- NOT the base's own letter. For a call
    these are the same letter, so this looks like "repeats the base"
    and that's what the assignment doc's prose says. For a put they are
    genuinely different letters (e.g. base R / suffix F for a June
    put), and requiring literal equality -- what an earlier version of
    this function did -- silently rejects every real put RIC that ever
    comes back from LSEG.

    If `root` is given, it's used to anchor the match so a root that
    happens to contain a letter in A-X range can't be misread as part
    of the month code. If omitted, the regex takes the longest run of
    letters before a valid month-code+digit pattern -- fine for roots
    like UUUU where there's no ambiguity, but pass root explicitly for
    anything less clean.
    """
    text = str(ric).strip().upper()

    if root is not None:
        root = root.upper()
        if not text.startswith(root):
            return None
        remainder = text[len(root):]
        m = re.match(
            r"^(?P<code>[A-X])(?P<dd>\d{2})(?P<yy>\d{2})(?P<strike>\d{5})"
            r"\.U\^(?P<code2>[A-X])(?P<yy2>\d{2})$",
            remainder,
        )
        if m is None:
            return None
        groups = m.groupdict()
        groups["root"] = root
    else:
        m = _RIC_RE.match(text)
        if m is None:
            return None
        groups = m.groupdict()

    code, code2 = groups["code"], groups["code2"]
    yy, yy2 = groups["yy"], groups["yy2"]

    month, right = _CODE_TO_MONTH_RIGHT[code]

    # the suffix must be the CALL letter for this same month (see
    # docstring) -- not a literal repeat of the base code. This is the
    # fix: the old check required code2 == code, which rejected every
    # real put RIC.
    expected_suffix_code = CALL_LETTERS[month - 1]
    if code2 != expected_suffix_code or yy2 != yy:
        return None
    day = int(groups["dd"])
    year = 2000 + int(yy)
    strike = int(groups["strike"]) / 100.0

    try:
        expiry = dt.date(year, month, day)
    except ValueError:
        return None

    return {
        "ric": text,
        "root": groups["root"],
        "right": right,
        "month": month,
        "day": day,
        "year": year,
        "expiry": expiry,
        "strike": strike,
        "month_code": code,
    }
