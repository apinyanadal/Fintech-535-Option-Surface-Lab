"""
test_ric_codec.py

No LSEG connection needed for any of this -- pure round-trip testing of
build_option_ric() / parse_option_ric() against each other.
"""

from ric_codec import build_option_ric, parse_option_ric


def test_roundtrip_basic_call():
    ric = build_option_ric("UUUU", month=1, day=15, year2=26, strike=12.50, right="C")
    parsed = parse_option_ric(ric, root="UUUU")
    assert parsed is not None, f"failed to parse {ric}"
    assert parsed["root"] == "UUUU"
    assert parsed["right"] == "C"
    assert parsed["expiry"].isoformat() == "2026-01-15"
    assert parsed["strike"] == 12.50


def test_roundtrip_put():
    ric = build_option_ric("UUUU", month=1, day=15, year2=26, strike=12.50, right="P")
    parsed = parse_option_ric(ric, root="UUUU")
    assert parsed["right"] == "P"
    assert parsed["month_code"] == "M"  # first put letter = Jan put


def test_roundtrip_all_twelve_months_both_rights():
    # every month code, both rights -- catches any off-by-one in the
    # letter tables before it touches real data
    for month in range(1, 13):
        for right in ("C", "P"):
            ric = build_option_ric("UUUU", month=month, day=20, year2=26,
                                    strike=9.00, right=right)
            parsed = parse_option_ric(ric, root="UUUU")
            assert parsed is not None, f"failed on month={month} right={right}"
            assert parsed["month"] == month
            assert parsed["right"] == right


def test_strike_padding_small_and_large():
    cases = [0.50, 5.00, 12.50, 99.99, 100.00]
    for strike in cases:
        ric = build_option_ric("UUUU", month=6, day=19, year2=26,
                                strike=strike, right="C")
        parsed = parse_option_ric(ric, root="UUUU")
        assert parsed is not None
        assert abs(parsed["strike"] - strike) < 1e-9, f"strike mismatch for {strike}"


def test_december_january_boundary():
    dec_ric = build_option_ric("UUUU", month=12, day=31, year2=25, strike=10.0, right="C")
    jan_ric = build_option_ric("UUUU", month=1, day=1, year2=26, strike=10.0, right="C")
    p_dec = parse_option_ric(dec_ric, root="UUUU")
    p_jan = parse_option_ric(jan_ric, root="UUUU")
    assert p_dec["expiry"].isoformat() == "2025-12-31"
    assert p_jan["expiry"].isoformat() == "2026-01-01"


def test_wrong_root_rejected():
    ric = build_option_ric("UUUU", month=1, day=15, year2=26, strike=12.5, right="C")
    assert parse_option_ric(ric, root="AAPL") is None


def test_garbage_rejected():
    assert parse_option_ric("not_a_ric") is None
    assert parse_option_ric("") is None
    assert parse_option_ric("UUUUZ1526012500.U^Z26") is None  # Z is not A-X


def test_mismatched_suffix_rejected():
    # hand-corrupt a valid RIC's suffix -- base says Jan call, suffix says Feb
    good = build_option_ric("UUUU", month=1, day=15, year2=26, strike=12.5, right="C")
    corrupted = good.replace("^A26", "^B26")
    assert parse_option_ric(corrupted, root="UUUU") is None


def test_put_suffix_uses_call_letter_not_put_letter():
    # the real behavior, confirmed against LSEG data: a June put's suffix
    # is F26 (June's CALL letter), not R26 (June's own put letter)
    ric = build_option_ric("UUUU", month=6, day=26, year2=26, strike=10.0, right="P")
    assert ric.endswith("^F26"), f"expected suffix ^F26, got {ric}"
    parsed = parse_option_ric(ric, root="UUUU")
    assert parsed is not None
    assert parsed["right"] == "P"
    assert parsed["month"] == 6


def test_real_lseg_rics_from_reference_pull():
    # actual RICs pulled from LSEG for UUUU -- the strongest possible
    # test, since these are real contracts, not our own synthetic ones.
    # A put's suffix uses the CALL letter for that month; a call's
    # suffix uses its own letter (which is also the call letter).
    real_call = "UUUUF122601100.U^F26"   # June call, day 12, strike 11.00
    real_put = "UUUUR122601200.U^F26"    # June put,  day 12, strike 12.00 -- note suffix F, not R

    p_call = parse_option_ric(real_call, root="UUUU")
    p_put = parse_option_ric(real_put, root="UUUU")

    assert p_call is not None, "real call RIC failed to parse"
    assert p_call["right"] == "C"
    assert p_call["strike"] == 11.00

    assert p_put is not None, "real put RIC failed to parse -- this was the actual bug"
    assert p_put["right"] == "P"
    assert p_put["strike"] == 12.00
    assert p_put["month"] == 6


def test_invalid_calendar_date_rejected():
    # Feb 30 doesn't exist -- dt.date() raising should surface as None,
    # not crash the caller
    ric = build_option_ric("UUUU", month=2, day=30, year2=26, strike=10.0, right="C")
    assert parse_option_ric(ric, root="UUUU") is None


def test_no_root_argument_still_works_for_clean_roots():
    ric = build_option_ric("UUUU", month=3, day=20, year2=26, strike=8.5, right="P")
    parsed = parse_option_ric(ric)  # no root hint
    assert parsed is not None
    assert parsed["root"] == "UUUU"
    assert parsed["strike"] == 8.5


if __name__ == "__main__":
    import sys
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
