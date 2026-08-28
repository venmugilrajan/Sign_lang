"""
Test Harness for Double Letter State Machine & Two-Tier Spellcheck Engine

Tests:
1. Double Letter Capture (e.g. 'HELLO' with double L, 'PLEASE' with double E)
2. Continuous Hold Spam Rejection (holding 'L' continuously only captures 1 'L')
3. Two-Tier Spellcheck:
   - Custom vocabulary direct & fuzzy match (e.g. 'HELO' -> 'HELLO', 'PLEAS' -> 'PLEASE')
   - Low vs High confidence weighting
   - General English dictionary fallback
   - Out-of-vocabulary graceful raw preservation + suggestions
"""

import sys
from word_buffer_engine import (
    WordBufferStateMachine,
    TwoTierSpellCorrector,
    levenshtein_distance,
    weighted_edit_distance
)

def run_tests():
    print("=" * 70)
    print("  [*] Running Sign Language Word Framing & Spellchecker Tests")
    print("=" * 70)

    corrector = TwoTierSpellCorrector()
    passed = 0
    total = 0

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: Double Letter Capture ('HELLO' with intentional release between L's)
    # ──────────────────────────────────────────────────────────────────────────
    total += 1
    sm = WordBufferStateMachine(corrector=corrector, stable_frames_needed=5, release_frames_needed=2)

    # Type 'H' (5 frames)
    for _ in range(5): sm.feed_frame('H', 0.95)
    # Type 'E' (5 frames)
    for _ in range(5): sm.feed_frame('E', 0.95)
    # Type first 'L' (5 frames)
    for _ in range(5): sm.feed_frame('L', 0.95)
    
    # Hand released / momentary pause (3 frames of NO HAND)
    for _ in range(3): sm.feed_frame('NO HAND', 0.0)
    
    # Type second 'L' (5 frames)
    for _ in range(5): sm.feed_frame('L', 0.95)
    # Type 'O' (5 frames)
    for _ in range(5): sm.feed_frame('O', 0.95)

    print(f"\nTest 1 - Repeated Letter Fingerspelling (H-E-L-release-L-O):")
    print(f"   Buffer: '{sm.current_word}' (Expected: 'HELLO')")
    if sm.current_word == "HELLO":
        print("   [+] PASSED: Second 'L' was successfully captured after gesture release.")
        passed += 1
    else:
        print(f"   [-] FAILED: Got '{sm.current_word}'")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Continuous Hold Spam Rejection (Holding 'A' for 30 frames)
    # ──────────────────────────────────────────────────────────────────────────
    total += 1
    sm.clear_all()
    for _ in range(30):
        sm.feed_frame('A', 0.95)

    print(f"\nTest 2 - Continuous Hold Spam Rejection (Hold 'A' for 30 frames):")
    print(f"   Buffer: '{sm.current_word}' (Expected: 'A')")
    if sm.current_word == "A":
        print("   [+] PASSED: Single 'A' captured, continuous hold spam was blocked.")
        passed += 1
    else:
        print(f"   [-] FAILED: Got '{sm.current_word}'")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3: Custom Vocabulary Spellcheck (Noisy Inputs)
    # ──────────────────────────────────────────────────────────────────────────
    test_cases = [
        # (input_raw, confidences, expected_output, description)
        ("HELLO", [0.9, 0.9, 0.9, 0.9, 0.9], "HELLO", "Exact custom match stays exact"),
        ("HELO", [0.9, 0.9, 0.8, 0.9], "HELLO", "Missing double letter corrected to HELLO"),
        ("PLEAS", [0.9, 0.9, 0.9, 0.9, 0.9], "PLEASE", "Truncated input corrected to PLEASE"),
        ("THSNK", [0.9, 0.9, 0.45, 0.9, 0.9], "THANKS", "Low-confidence letter 'S' fixed to 'A' (THANKS)"),
        ("SRRY", [0.9, 0.9, 0.9, 0.9], "SORRY", "Missing letter fixed to custom word SORRY"),
        ("XYZQW", [0.9, 0.9, 0.9, 0.9, 0.9], "XYZQW", "Unmatched junk preserved without wrong forced correction"),
    ]

    print(f"\nTest 3 - Two-Tier Spellchecker Custom Vocabulary & Confidence Weighting:")
    for raw_in, confs, exp_out, desc in test_cases:
        total += 1
        res, is_corr, suggs = corrector.correct(raw_in, confs)
        ok = (res == exp_out)
        if ok:
            passed += 1
            print(f"   [+] [{desc}]: '{raw_in}' -> '{res}' (Suggestions: {suggs})")
        else:
            print(f"   [-] [{desc}]: '{raw_in}' -> Got '{res}', Expected '{exp_out}'")

    print(f"\n{'='*70}")
    print(f"  Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
