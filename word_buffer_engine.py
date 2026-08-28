"""
Sign Language Word Framing, Custom Vocabulary & Two-Tier Spellcheck Engine

Features:
1. Short Word Protection (Common short words like 'HI', 'NO', 'OK', 'YES', 'ME', 'MY', 'GO' preserved as-is)
2. Custom Sign Vocabulary with domain-specific priority
3. Edit Distance Capped for Short Words (length <= 3 words strictly cap edit distance to <= 1)
4. Confidence-weighted Levenshtein penalty for longer words
5. State Machine for Gesture Release Detection (enables repeated letters like 'HELLO', 'PLEASE')
6. Detailed logging after each buffer append
"""

import numpy as np
from spellchecker import SpellChecker

# Standard short English sign words that should NEVER be aggressively overridden by edit distance
SHORT_VALID_WORDS = {
    "HI", "NO", "YES", "OK", "GO", "ME", "MY", "HE", "WE", "US", "AM", "IS",
    "IN", "ON", "AT", "TO", "DO", "SO", "IF", "UP", "BY", "AN", "AS", "IT",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
    "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
    "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
    "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE"
}

# Default Custom Sign Vocabulary
DEFAULT_SIGN_VOCABULARY = [
    "HI", "HELLO", "THANKYOU", "THANKS", "PLEASE", "YES", "NO", "SORRY", "WELCOME",
    "GOOD", "BAD", "NAME", "HELP", "HOW", "WHAT", "WHERE", "WHEN", "WHY",
    "WHO", "FINE", "NICE", "MEET", "YOU", "MY", "ME", "SIGN", "LANGUAGE",
    "LEARN", "DEAF", "FRIEND", "LOVE", "HAPPY", "SAD", "WATER", "FOOD",
    "EAT", "DRINK", "STOP", "GO", "TIME", "TODAY", "TOMORROW", "YESTERDAY"
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def weighted_edit_distance(raw_word: str, target_word: str, letter_confidences: list = None) -> float:
    """
    Computes edit distance weighted by model letter confidence.
    Changes on low-confidence letters are penalized less than changes on high-confidence letters.
    """
    if not letter_confidences or len(letter_confidences) != len(raw_word):
        return float(levenshtein_distance(raw_word, target_word))

    m, n = len(raw_word), len(target_word)
    dp = np.zeros((m + 1, n + 1), dtype=np.float32)

    for i in range(m + 1):
        dp[i][0] = i * 1.0
    for j in range(n + 1):
        dp[0][j] = j * 1.0

    for i in range(1, m + 1):
        char_conf = letter_confidences[i - 1]
        sub_cost = 1.2 if char_conf > 0.85 else (0.6 if char_conf < 0.60 else 1.0)
        del_cost = 1.2 if char_conf > 0.85 else 0.8

        for j in range(1, n + 1):
            if raw_word[i - 1] == target_word[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + del_cost,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + sub_cost
                )

    return float(dp[m][n])


class TwoTierSpellCorrector:
    """
    Two-tier word corrector for sign language letter buffers:
    Tier 0: Short word protection (words <= 3 chars that are valid words stay as-is).
    Tier 1: Custom Sign Vocabulary matching via weighted Levenshtein distance.
            - length <= 3: max edit distance allowed is 1
            - length > 3: max edit distance allowed is 2
    Tier 2: General English dictionary (pyspellchecker) fallback.
    Tier 3: Raw preservation with 'did you mean' suggestions if no direct match exists.
    """
    def __init__(self, custom_vocab: list = None):
        self.custom_vocab = [w.upper() for w in (custom_vocab or DEFAULT_SIGN_VOCABULARY)]
        self.general_spell = SpellChecker()

    def add_words(self, words: list):
        """Add additional words to custom vocabulary."""
        for w in words:
            w_clean = w.strip().upper()
            if w_clean and w_clean not in self.custom_vocab:
                self.custom_vocab.append(w_clean)

    def correct(self, raw_word: str, letter_confidences: list = None):
        """
        Corrects a raw word buffer into the most accurate target sign word.
        Returns: (final_word, is_corrected, suggestions_list)
        """
        raw = raw_word.strip().upper()
        if not raw:
            return "", False, []

        # 0. Short valid words check (e.g. 'HI', 'NO', 'OK', 'YES', 'ME', 'MY', 'GO')
        if raw in SHORT_VALID_WORDS or raw in self.custom_vocab:
            return raw, False, []

        # Strict edit distance threshold based on word length:
        # 1-2 letter words: max dist 0 (must be exact)
        # 3 letter words: max dist 1
        # 4+ letter words: max dist 2
        if len(raw) <= 2:
            max_allowed_dist = 0
        elif len(raw) == 3:
            max_allowed_dist = 1
        else:
            max_allowed_dist = 2

        # Tier 1: Custom Sign Vocabulary match
        vocab_matches = []
        if max_allowed_dist > 0:
            for target in self.custom_vocab:
                raw_dist = levenshtein_distance(raw, target)
                if raw_dist <= max_allowed_dist:
                    w_dist = weighted_edit_distance(raw, target, letter_confidences)
                    vocab_matches.append((target, raw_dist, w_dist))

        if vocab_matches:
            vocab_matches.sort(key=lambda x: (x[2], x[1], abs(len(x[0]) - len(raw))))
            best_match = vocab_matches[0][0]
            suggestions = [m[0] for m in vocab_matches[:3]]
            return best_match, (best_match != raw), suggestions

        # Tier 2: General English Dictionary fallback (only if allowed dist > 0)
        if max_allowed_dist > 0:
            gen_corr = self.general_spell.correction(raw.lower())
            if gen_corr:
                gen_corr_upper = gen_corr.upper()
                if levenshtein_distance(raw, gen_corr_upper) <= max_allowed_dist:
                    return gen_corr_upper, True, [gen_corr_upper]

        # Tier 3: No close match found — preserve raw buffer and suggest nearest custom words
        all_candidates = []
        for target in self.custom_vocab:
            dist = levenshtein_distance(raw, target)
            if dist <= 3:
                all_candidates.append((target, dist))

        all_candidates.sort(key=lambda x: x[1])
        suggestions = [c[0] for c in all_candidates[:3]]

        return raw, False, suggestions


class WordBufferStateMachine:
    """
    State machine managing real-time letter stream, release detection,
    double-letter handling ('HELLO', 'PLEASE'), and word boundary commits.
    """
    def __init__(self, corrector: TwoTierSpellCorrector = None,
                 stable_frames_needed: int = 8,
                 release_frames_needed: int = 2):
        self.corrector = corrector or TwoTierSpellCorrector()
        self.stable_frames_needed = stable_frames_needed
        self.release_frames_needed = release_frames_needed

        # Buffer state
        self.word_letters = []       # list of confirmed chars ['H', 'E', 'L', 'L', 'O']
        self.letter_confidences = [] # list of floats [0.92, 0.88, ...]
        self.sentence = []           # list of committed words

        # State tracking for repeated letter detection
        self.last_confirmed_letter = ""
        self.released_since_last_confirm = True
        self.release_counter = 0

        # Current candidate stream
        self.candidate_letter = ""
        self.candidate_count = 0

    @property
    def current_word(self) -> str:
        return "".join(self.word_letters)

    def feed_frame(self, raw_pred: str, conf: float, min_conf: float = 0.65):
        """
        Feeds a live raw frame prediction into the state machine.
        Returns: dict of state updates
        """
        captured_letter = None

        if raw_pred is None or raw_pred == "NO HAND" or conf < min_conf:
            self.release_counter += 1
            if self.release_counter >= self.release_frames_needed:
                self.released_since_last_confirm = True

            self.candidate_letter = ""
            self.candidate_count = 0
            return self._get_status(captured_letter)

        if raw_pred != self.last_confirmed_letter:
            self.release_counter += 1
            if self.release_counter >= self.release_frames_needed:
                self.released_since_last_confirm = True
        else:
            self.release_counter = 0

        # Stability counter
        if raw_pred == self.candidate_letter:
            self.candidate_count += 1
        else:
            self.candidate_letter = raw_pred
            self.candidate_count = 1

        # Check if stability threshold reached
        if self.candidate_count >= self.stable_frames_needed:
            is_same_as_last = (raw_pred == self.last_confirmed_letter)

            # ALLOW confirm if it's a new letter OR if hand was released since last confirm
            if (not is_same_as_last) or self.released_since_last_confirm:
                captured_letter = self._confirm_letter(raw_pred, conf)

        return self._get_status(captured_letter)

    def _confirm_letter(self, letter: str, conf: float) -> str:
        """Applies confirmed gesture action with detailed console logging."""
        l_lower = letter.lower()
        self.last_confirmed_letter = letter
        self.released_since_last_confirm = False
        self.release_counter = 0
        self.candidate_count = 0

        if l_lower in ["space", "_"]:
            print(f"[WORD BUFFER] Gesture 'SPACE' confirmed -> Committing word '{self.current_word}'")
            self.commit_word()
            return "SPACE"
        elif l_lower in ["del", "delete"]:
            if self.word_letters:
                popped = self.word_letters.pop()
                if self.letter_confidences: self.letter_confidences.pop()
                print(f"[WORD BUFFER] Gesture 'DEL' confirmed -> Removed '{popped}' | Buffer now: {self.word_letters}")
            return "DEL"
        elif l_lower == "nothing":
            return "NOTHING"
        else:
            char_upper = letter.upper()
            self.word_letters.append(char_upper)
            self.letter_confidences.append(conf)
            print(f"[WORD BUFFER] Confirmed '{char_upper}' (conf: {conf*100:.1f}%) | Buffer after append: {self.word_letters} -> '{self.current_word}'")
            return char_upper

    def commit_word(self):
        """Commits current word buffer into sentence using two-tier spellchecker."""
        raw_word = self.current_word.strip()
        if not raw_word:
            return

        final_word, is_corr, suggestions = self.corrector.correct(raw_word, self.letter_confidences)
        print(f"[WORD COMMIT] Raw: '{raw_word}' -> Final: '{final_word}' (Corrected: {is_corr}) | Sentence: {self.sentence + [final_word]}")
        self.sentence.append(final_word)

        # Reset word buffer
        self.word_letters.clear()
        self.letter_confidences.clear()
        self.last_confirmed_letter = ""
        self.released_since_last_confirm = True
        self.candidate_letter = ""
        self.candidate_count = 0

    def backspace(self):
        """Deletes last letter from buffer or pops last word from sentence."""
        if self.word_letters:
            popped = self.word_letters.pop()
            if self.letter_confidences:
                self.letter_confidences.pop()
            print(f"[WORD BUFFER] Manual Backspace -> Removed '{popped}' | Buffer now: {self.word_letters}")
        elif self.sentence:
            popped_word = self.sentence.pop()
            print(f"[WORD BUFFER] Manual Backspace -> Removed word '{popped_word}' from sentence")

    def clear_all(self):
        """Clears everything."""
        self.word_letters.clear()
        self.letter_confidences.clear()
        self.sentence.clear()
        self.last_confirmed_letter = ""
        self.released_since_last_confirm = True
        self.candidate_letter = ""
        self.candidate_count = 0
        print("[WORD BUFFER] Cleared all buffers and sentence.")

    def _get_status(self, captured_letter=None) -> dict:
        curr = self.current_word
        suggested, is_corr, sugg_list = self.corrector.correct(curr, self.letter_confidences) if curr else ("", False, [])
        return {
            "captured_letter": captured_letter,
            "word_buffer": curr,
            "sentence": " ".join(self.sentence),
            "suggested_word": suggested,
            "suggestions": sugg_list,
            "hold_progress": min(1.0, self.candidate_count / float(self.stable_frames_needed)),
            "released": self.released_since_last_confirm
        }
