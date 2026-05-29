import os
import re
import json
import requests
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# ==============================
# Load API Key
# ==============================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ==============================
# Cleaning + helpers
# ==============================
def clean_text(text):
    text = str(text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # remove tashkeel
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_mixed_arabic_letters(text: str) -> str:
    """
    Fix rare LLM mistakes like:
    Bنك مصر -> بنك مصر
    """
    replacements = {
        "B": "ب", "b": "ب",
        "A": "ا", "a": "ا",
        "K": "ك", "k": "ك",
        "L": "ل", "l": "ل",
        "M": "م", "m": "م",
        "N": "ن", "n": "ن",
        "R": "ر", "r": "ر",
        "S": "س", "s": "س",
        "T": "ت", "t": "ت",
        "W": "و", "w": "و",
        "Y": "ي", "y": "ي",
        "H": "ه", "h": "ه",
    }
    text = str(text)
    for eng, ar in replacements.items():
        text = text.replace(eng, ar)
    return text


def is_single_arabic_letter(token: str) -> bool:
    """
    True for one Arabic letter only:
    ا / ب / ت / م / ه ...
    """
    token = clean_text(token)
    return bool(re.fullmatch(r"[\u0621-\u064A]", token))


def has_english_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text)))


def is_number_token(token: str) -> bool:
    return clean_text(token).isdigit()


def all_tokens_are_single_letters(tokens) -> bool:
    """
    Prevent matching ["ه", "ش", "ا", "م"] as a fuzzy phrase.
    """
    return bool(tokens) and all(is_single_arabic_letter(t) for t in tokens)


def normalize_for_compare(text: str) -> str:
    """
    Used only for comparison, not final output.
    Final output should still use the original dataset word.
    """
    text = clean_text(text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text


# ==============================
# Dataset Matcher
# ==============================
class DatasetMatcher:
    def __init__(self, dataset_items):
        self.original = [clean_text(w) for w in dataset_items if clean_text(w)]

        # remove duplicates while preserving order
        self.original = list(dict.fromkeys(self.original))

        # exact lookup
        self.lookup = {clean_text(w): w for w in self.original}

        # normalized lookup for light Arabic variations
        self.normalized_lookup = {}
        for w in self.original:
            self.normalized_lookup.setdefault(normalize_for_compare(w), w)

        self.normalized = [clean_text(w) for w in self.original]

    def exact(self, text):
        text = clean_text(fix_mixed_arabic_letters(text))

        # exact first
        if text in self.lookup:
            return self.lookup[text]

        # normalized exact second
        n_text = normalize_for_compare(text)
        return self.normalized_lookup.get(n_text)

    def can_use_fuzzy(self, text):
        """
        Very important:
        Fuzzy matching caused:
        احمد -> الحمد لله
        and letters being merged into wrong words.

        So fuzzy is blocked for:
        - single Arabic letters
        - numbers
        - short words
        - text with English letters
        """
        text = clean_text(text)

        if not text:
            return False

        if has_english_letters(text):
            return False

        if is_single_arabic_letter(text):
            return False

        if is_number_token(text):
            return False

        # short words are dangerous with fuzzy
        if len(text.replace(" ", "")) <= 3:
            return False

        return True

    def fuzzy(self, text, threshold=93):
        """
        Controlled fuzzy only.
        Higher threshold prevents احمد -> الحمد لله.
        """
        text = clean_text(fix_mixed_arabic_letters(text))

        if not self.can_use_fuzzy(text):
            return None

        result = process.extractOne(
            text,
            self.normalized,
            scorer=fuzz.ratio
        )

        if not result:
            return None

        match, score, index = result
        candidate = self.original[index]

        # Extra protection:
        # avoid matching a one-word token to a long phrase unless score is extremely high.
        if len(text.split()) == 1 and len(candidate.split()) > 1 and score < 98:
            return None

        # Extra protection:
        # avoid very different length matches
        text_len = len(text.replace(" ", ""))
        cand_len = len(candidate.replace(" ", ""))
        if abs(text_len - cand_len) > max(2, int(text_len * 0.5)) and score < 98:
            return None

        if score >= threshold:
            return candidate

        return None

    def char_level(self, word):
        """
        Convert unknown words/names to letters.
        Only keep letters that exist in dataset.
        """
        word = clean_text(fix_mixed_arabic_letters(word))

        chars = []
        for ch in word:
            if not ch.strip():
                continue

            match = self.exact(ch)

            if match:
                chars.append(match)
            else:
                # If the letter is not in dataset, ignore it instead of inventing.
                # You can append ch if you want debugging:
                # chars.append(ch)
                pass

        return chars


# ==============================
# Main Pipeline
# ==============================
class ArabicNormalizer:

    def __init__(self):
        self.url = "https://openrouter.ai/api/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        with open("poses.json", "r", encoding="utf-8") as f:
            pose_db = json.load(f)

        self.pose_db = set(pose_db.keys())

        # cleaning + preparing
        self.pose_db = [clean_text(w) for w in self.pose_db if w]
        self.pose_db = list(dict.fromkeys(self.pose_db))

        # longer phrases first
        self.pose_db.sort(key=lambda x: len(x.split()), reverse=True)

        self.matcher = DatasetMatcher(self.pose_db)

    # ==============================
    # LLM STEP
    # ==============================
    def _build_dataset_summary(self):
        """
        Split the dataset into categories so the LLM can reason about
        each type independently rather than scanning one flat list.
        """
        import re as _re
        phrases     = [w for w in self.pose_db if " " in w]
        single_words= [w for w in self.pose_db if " " not in w
                       and not w.isdigit()
                       and not _re.fullmatch(r"[\u0621-\u064A]", w)]
        letters     = [w for w in self.pose_db if _re.fullmatch(r"[\u0621-\u064A]", w)]
        numbers     = [w for w in self.pose_db if w.isdigit()]
        return phrases, single_words, letters, numbers

    def normalize(self, sentence):
        phrases, single_words, letters, numbers = self._build_dataset_summary()

        system_prompt = """\
You are an expert Arabic Sign Language (لغة الإشارة العربية) linguist and NLP system.
Your ONLY job is to map any Arabic or Egyptian-dialect sentence to tokens that exist \
in the provided Sign Language Dataset, then output a JSON list — nothing else.

════════════════════════════════════════
OUTPUT FORMAT  (NON-NEGOTIABLE)
════════════════════════════════════════
• Return EXACTLY one JSON array of strings, e.g.  ["انا", "يريد", "ماء"]
• No markdown fences. No explanation. No transliteration. No English letters.
• All strings must be pure Arabic Unicode. Never mix Latin letters into Arabic words.

════════════════════════════════════════
STEP-BY-STEP REASONING PROCESS
════════════════════════════════════════
Before writing the final list, silently work through these steps in order:

STEP 1 – CLEAN INPUT
  Remove diacritics (tashkeel). Normalise alef variants (إأآٱ → ا), ta-marbuta (ة → ه \
only for matching, keep original form in output). Strip punctuation.

STEP 2 – DIALECT → MSA CONVERSION
  Convert Egyptian/colloquial Arabic words to their Modern Standard Arabic (MSA) equivalents \
before looking them up. Examples (not exhaustive — reason by meaning):
    ماما/أمي → أم          بابا/والدي → أب
    عايز/عاوز/بدي → يريد   بحب/بعشق  → يحب
    اروح/رايح/يمشي → يذهب  جاي/رايح  → يجيء / يذهب
    دكتور → طبيب           تعبان/وجعان/تعبانة → مريض
    كورة → كرة             ميه/مية → ماء
    الشغل/الوظيفة → عمل    صاحبي/صحبي → صديق
    اخويا/اخوي → أخ        اختي → أخت
    جنيه/فلوس → مال        عربية/سيارة → سيارة
    مدرسة/المدرسة → مدرسة  أوضة/غرفة → غرفة
    سما/ريحة → سماء/هواء   زي/مثل → مثل
    كمان/برضو → أيضا       دلوقتي/هلق → الآن
  For any dialect word not in the list above, think about its MSA meaning and look up that meaning.

STEP 3 – GREEDY PHRASE MATCHING (longest first)
  Scan from left to right. At each position try to match the longest possible
  sequence of consecutive words (up to 6) against the MULTI-WORD PHRASES list.
  If matched: emit the phrase, advance past those words, continue.

STEP 4 – SINGLE-WORD MATCHING
  For each remaining word, look it up in the SINGLE WORDS list (exact or near-exact \
after alef/ya/ta-marbuta normalisation). If matched, emit it.

STEP 5 – NAMES & UNKNOWN WORDS → LETTER-SPELL
  If a word is a proper name or cannot be matched in any list:
  • Split it into individual Arabic letters.
  • Only emit letters that appear in the LETTERS list.
  • NEVER substitute a name with a semantically similar word or phrase.
  BAD:  احمد → الحمد لله        GOOD (if احمد not in dataset): احمد → ["ا","ح","م","د"]

STEP 6 – NUMBERS
  If the number exists in NUMBERS list, emit it.
  Otherwise decompose place-by-place using available numbers:
    46  → ["40","6"]      125 → ["100","20","5"]      999 → ["900","90","9"]

STEP 7 – FINAL CHECK
  • Every meaningful input word must appear in the output (or be letter-spelled).
  • Single letters must never be fuzzy-matched to longer words.
  • No item in the output may contain Latin characters.
  • Return ONLY the JSON array."""

        user_prompt = f"""\
════════════════════════════════════════
DATASET
════════════════════════════════════════

MULTI-WORD PHRASES ({len(phrases)} entries — prefer these first):
{json.dumps(phrases, ensure_ascii=False)}

SINGLE WORDS ({len(single_words)} entries):
{json.dumps(single_words, ensure_ascii=False)}

AVAILABLE ARABIC LETTERS (for spelling unknown words/names):
{json.dumps(letters, ensure_ascii=False)}

AVAILABLE NUMBERS:
{json.dumps(numbers, ensure_ascii=False)}

════════════════════════════════════════
SENTENCE TO CONVERT
════════════════════════════════════════
{sentence}

════════════════════════════════════════
OUTPUT  (JSON array only — no other text)
════════════════════════════════════════"""

        data = {
            "model": "openai/gpt-4o-mini",
            "temperature": 0.0,          # deterministic — we don't want creative guessing
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        }

        res = requests.post(self.url, headers=self.headers, json=data)

        if res.status_code != 200:
            raise Exception(res.text)

        return res.json()["choices"][0]["message"]["content"].strip()

    # ==============================
    # Parse LLM Output Safely
    # ==============================
    def parse_llm_output(self, llm_output):
        llm_output = fix_mixed_arabic_letters(llm_output)

        try:
            tokens = json.loads(llm_output)
            if isinstance(tokens, list):
                return [clean_text(fix_mixed_arabic_letters(t)) for t in tokens if clean_text(t)]
        except Exception:
            pass

        # fallback parser
        tokens = [
            t.strip(" []،,\"'")
            for t in llm_output.replace(",", " ").split()
        ]
        return [clean_text(fix_mixed_arabic_letters(t)) for t in tokens if clean_text(t)]

    # ==============================
    # CORE MAPPING - FIXED
    # ==============================
    def map_to_dataset(self, tokens):
        """
        Fixed logic:
        - No full sentence fuzzy.
        - No fuzzy for single letters.
        - No fuzzy for unknown names/short words.
        - If unknown word/name, convert to letters.
        - Phrase fuzzy is blocked when phrase consists of single letters.
        """

        tokens = [clean_text(fix_mixed_arabic_letters(t)) for t in tokens if clean_text(t)]

        mapped = []
        i = 0
        n = len(tokens)

        while i < n:
            found = None
            best_len = 0

            # ------------------------------
            # 1) PHRASE MATCH
            # exact phrase first, fuzzy phrase only when safe
            # ------------------------------
            max_phrase_len = min(6, n - i)

            for j in range(max_phrase_len, 1, -1):
                phrase_tokens = tokens[i:i + j]
                phrase = " ".join(phrase_tokens)

                # exact phrase is always safe
                match = self.matcher.exact(phrase)

                # fuzzy phrase is NOT safe for sequences of letters
                if not match and not all_tokens_are_single_letters(phrase_tokens):
                    match = self.matcher.fuzzy(phrase, threshold=95)

                if match:
                    found = match
                    best_len = j
                    break

            if found:
                mapped.append(found)
                i += best_len
                continue

            # ------------------------------
            # 2) WORD MATCH
            # exact only first
            # ------------------------------
            word = tokens[i]

            exact_match = self.matcher.exact(word)
            if exact_match:
                mapped.append(exact_match)
                i += 1
                continue

            # ------------------------------
            # 3) SINGLE LETTER
            # Never fuzzy match a single letter.
            # ------------------------------
            if is_single_arabic_letter(word):
                letter_match = self.matcher.exact(word)
                if letter_match:
                    mapped.append(letter_match)
                i += 1
                continue

            # ------------------------------
            # 4) UNKNOWN WORD / NAME
            # Convert to letters BEFORE fuzzy.
            # This prevents احمد -> الحمد لله.
            # ------------------------------
            chars = self.matcher.char_level(word)
            if chars:
                print(f"⚠️ Char-level fallback: {word} -> {chars}")
                mapped.extend(chars)
                i += 1
                continue

            # ------------------------------
            # 5) Controlled fuzzy as LAST resort only
            # ------------------------------
            fuzzy_match = self.matcher.fuzzy(word, threshold=95)
            if fuzzy_match:
                mapped.append(fuzzy_match)
                i += 1
                continue

            print(f"⚠️ No match found, skipped: {word}")
            i += 1

        return mapped

    # ==============================
    # RUN PIPELINE
    # ==============================
    def tokenize(self, sentence):
        print(f"\n🟡 Input: {sentence}")

        llm_output = self.normalize(sentence)
        print(f"🔵 LLM Output: {llm_output}")

        tokens = self.parse_llm_output(llm_output)
        print(f"🧩 Parsed Tokens: {tokens}")

        result = self.map_to_dataset(tokens)

        print(f"🟢 Final Output: {result}")

        return result


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    normalizer = ArabicNormalizer()

    sentence = "قطر"

    normalizer.tokenize(sentence)