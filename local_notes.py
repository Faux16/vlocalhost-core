"""Meeting notes with no model at all.

:mod:`summarizer` writes the good notes, and needs Ollama plus a pulled model
to do it. That is a separate application to install and a multi-gigabyte
download, which is a lot to ask of someone who only wanted their meeting
written down — and when it is missing the old behaviour was to save the raw
transcript and report a failure. A recording that produced nothing but a wall
of timestamps is a bad outcome for a reason the user may not be able to fix.

So this module writes notes from the transcript alone: no model, no download,
no network, works on a machine that has never been online. It cannot write
prose, because nothing here understands anything. What it can do is *find* —
score sentences by how much of the meeting's own vocabulary they carry, and
pick out the ones that look like a decision, a task, or a question. Every line
it emits was said by someone in the room.

Two entry points, used differently:

:func:`summarize`
    Stand-in notes, in the same sections :mod:`summarizer` produces, for when
    the model cannot be reached.

:func:`facts`
    Counted things — who spoke how much, how long it ran, what was asked.
    Appended to *every* set of notes, including the model's, because these are
    measured rather than generated and so are the one part that cannot be
    wrong in the way a language model is wrong.
"""

import re
from collections import Counter

# Words too common to say anything about what a meeting was about. Kept short
# on purpose: this is for ranking sentences, not for linguistics, and a longer
# list mostly adds words that never survive the frequency cut anyway.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he her here hers herself him himself
his how i i'd i'll i'm i've if in into is isn't it it's its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
this those through to too under until up very was wasn't we were weren't what
when where which while who whom why with won't would wouldn't you your yours
yourself yourselves just really actually basically kind sort like okay ok yeah
yes um uh right well going get got think know mean thing things lot bit
""".split())

# A sentence that looks like something was settled.
_DECISION_CUES = (
    "we decided", "we've decided", "we have decided", "decision is",
    "the decision", "we agreed", "we've agreed", "agreed to", "agreed that",
    "let's go with", "we'll go with", "we will go with", "signed off",
    "sign off on", "approved", "we're going with", "settled on",
    "made the call", "final answer", "that's confirmed",
)

# A sentence that looks like somebody now owes somebody something.
_ACTION_CUES = (
    "i'll", "i will", "we'll", "we will", "let's", "lets ", "we should",
    "you should", "we need to", "i need to", "you need to", "needs to",
    "can you", "could you", "please ", "action item", "to-do", "todo",
    "follow up", "follow-up", "take care of", "i'm going to",
    "we're going to", "make sure", "by monday", "by tuesday", "by wednesday",
    "by thursday", "by friday", "by tomorrow", "by next week", "by the end of",
    "deadline",
)

_LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*)$")
# The " (en)" the transcriber appends when detecting per utterance.
_LANG_TAG = re.compile(r"\s*\([a-z]{2,3}\)$", re.I)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


class Utterance:
    """One transcript line, split back into its parts."""

    __slots__ = ("time", "speaker", "text")

    def __init__(self, time, speaker, text):
        self.time = time
        self.speaker = speaker
        self.text = text

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<{self.time} {self.speaker or '-'}: {self.text[:40]!r}>"


def _split_speaker(rest):
    """Separate "You: hello" into ("You", "hello").

    A mic-only transcript carries no speaker label at all, and its text is
    free to contain a colon — "the plan is this: we ship Friday" must not be
    read as a speaker called "the plan is this". Length alone doesn't separate
    those, so the test is that a label *looks like a name*: a word or three,
    each capitalised, optionally followed by the language tag the transcriber
    adds. Written labels are "You" and "Participants" (config.LABEL_ME and
    LABEL_THEM), possibly renamed, so matching shape rather than exact text
    keeps working when they are.
    """
    head, sep, tail = rest.partition(":")
    if not sep or not tail.strip():
        return "", rest.strip()
    name = _LANG_TAG.sub("", head).strip()
    words = name.split()
    if (not words or len(words) > 3 or len(name) > 40
            or re.search(r"[.!?,;]", name)
            or not all(word[:1].isupper() for word in words)):
        return "", rest.strip()
    return head.strip(), tail.strip()


def parse(transcript):
    """Transcript text -> [Utterance].

    Live transcripts are entirely timestamped lines. A transcript read back
    off disk has a title and a "Saved:" banner on the front, and counting
    those as things somebody said inflates every tally below — so once any
    timestamped line is found, un-timestamped ones are treated as chrome. A
    transcript with no timestamps at all is taken at face value, since then
    the plain lines are all there is.
    """
    lines = [raw.strip() for raw in transcript.splitlines() if raw.strip()]
    matches = [(line, _LINE.match(line)) for line in lines]
    timestamped = any(m for _line, m in matches)

    out = []
    for line, match in matches:
        if match:
            stamp, rest = match.groups()
            speaker, text = _split_speaker(rest)
            if text:
                out.append(Utterance(stamp, speaker, text))
        elif not timestamped and not line.startswith(("=", "#", "Saved:")):
            out.append(Utterance("", "", line))
    return out


def _sentences(utterances):
    """[(utterance, sentence)] — utterances split into sentences, in order."""
    pairs = []
    for utt in utterances:
        for piece in re.split(r"(?<=[.!?])\s+", utt.text):
            piece = piece.strip()
            if piece:
                pairs.append((utt, piece))
    return pairs


def _words(text):
    return [w.lower() for w in _WORD.findall(text)]


def _content_words(text):
    return [w for w in _words(text) if w not in _STOPWORDS and len(w) > 2]


def _frequencies(pairs):
    counts = Counter()
    for _utt, sentence in pairs:
        counts.update(_content_words(sentence))
    return counts


def _score(sentence, freq):
    """How much of the meeting's own vocabulary this sentence carries.

    Distinct words only, so repeating one word does not win, and divided by
    length so a rambling sentence does not beat a dense one purely on size.
    """
    words = set(_content_words(sentence))
    if len(words) < 3:
        return 0.0
    return sum(freq[w] for w in words) / (len(words) ** 0.5)


def _from_cue(sentence, cues, limit=32):
    """The part of `sentence` that starts at the cue, or None if no cue.

    Whisper punctuates loosely, so an "utterance" is often several thoughts
    run together — "…a great journey so far Now let's listen to one song".
    Matching "let's" and then quoting the whole run-on presents thirty words
    of preamble as the task. Starting at the cue keeps the part that is
    actually the task, and the word cap stops a sentence that never ends from
    filling the section.
    """
    low = sentence.lower()
    at = min((low.find(cue) for cue in cues if cue in low), default=-1)
    if at < 0:
        return None
    clipped = sentence[at:].strip(" ,;-")
    words = clipped.split()
    if len(words) > limit:
        clipped = " ".join(words[:limit]) + "…"
    return clipped[:1].upper() + clipped[1:] if clipped else None


def _cued(pairs, cues, limit):
    """[(utterance, text)] for sentences carrying one of `cues`, deduplicated."""
    found, seen = [], set()
    for utt, sentence in pairs:
        text = _from_cue(sentence, cues)
        if not text or text.lower() in seen or len(text.split()) < 3:
            continue
        seen.add(text.lower())
        found.append((utt, text))
        if len(found) >= limit:
            break
    return found


def _ranked(pairs, freq, limit, skip=()):
    """The `limit` highest-scoring sentences, returned in the order spoken."""
    seen = set(skip)
    scored = []
    for index, (utt, sentence) in enumerate(pairs):
        key = sentence.lower()
        if key in seen or len(sentence.split()) < 5:
            continue
        seen.add(key)
        scored.append((_score(sentence, freq), index, utt, sentence))
    scored.sort(key=lambda row: row[0], reverse=True)
    top = sorted(scored[:limit], key=lambda row: row[1])
    return [(utt, sentence) for _s, _i, utt, sentence in top]


def _attribute(utt):
    """" — You, at 10:04:12" for a line we know the speaker or time of."""
    bits = []
    if utt.speaker:
        bits.append(utt.speaker)
    if utt.time:
        bits.append(utt.time)
    return f" — {', '.join(bits)}" if bits else ""


def title(transcript, words=4):
    """A few words naming the meeting, for the file name. '' if too thin.

    :mod:`summarizer` asks the model for this; without one, the most
    distinctive words in the room beat a bare timestamp.
    """
    freq = _frequencies(_sentences(parse(transcript)))
    # Too little was said for any word to mean anything; a timestamp is a
    # more honest file name than two words plucked out of forty.
    if sum(freq.values()) < 40:
        return ""
    # A word said once is noise; a word said repeatedly is the topic.
    common = [w for w, n in freq.most_common(words * 3) if n > 1]
    if len(common) < 2:
        return ""
    return " ".join(common[:words])


def summarize(transcript):
    """Markdown notes built only from what was said. Never raises.

    Mirrors the sections :func:`summarizer.summarize` produces, so a reader
    does not have to learn a second shape of document, and says plainly at the
    top which one they are looking at.
    """
    utterances = parse(transcript)
    pairs = _sentences(utterances)
    if not pairs:
        return ("## Summary\n\nNothing was transcribed, so there is nothing "
                "to summarize.\n")

    freq = _frequencies(pairs)
    lines = [
        "> **Built-in notes.** Written from the transcript on this machine, "
        "with no language model: the lines below were selected from what was "
        "said, not composed. Install [Ollama](https://ollama.com/download) "
        "for written notes.",
        "",
        "## Summary",
        "",
    ]

    headline = _ranked(pairs, freq, 3)
    if headline:
        lines.append(" ".join(sentence for _utt, sentence in headline))
    else:
        lines.append("The meeting was too short to pick anything out of.")
    used = {sentence.lower() for _utt, sentence in headline}

    lines += ["", "## Key Discussion Points", ""]
    points = _ranked(pairs, freq, 6, skip=used)
    if points:
        lines += [f"- {sentence}{_attribute(utt)}" for utt, sentence in points]
        used |= {sentence.lower() for _utt, sentence in points}
    else:
        lines.append("- Nothing beyond the summary above.")

    lines += ["", "## Decisions", ""]
    decisions = _cued(pairs, _DECISION_CUES, 6)
    if decisions:
        lines += [f"- {text}{_attribute(utt)}" for utt, text in decisions]
    else:
        lines.append("None recorded.")

    lines += ["", "## Action Items", ""]
    actions = _cued(pairs, _ACTION_CUES, 8)
    if actions:
        lines += [f"- [ ] {text}{_attribute(utt)}" for utt, text in actions]
    else:
        lines.append("None recorded.")

    return "\n".join(lines) + "\n"


def facts(transcript):
    """Counted things about the meeting — never inferred, so never wrong.

    Appended to the model's notes as well as the built-in ones. A summary can
    be confidently mistaken about who said what; a tally cannot.
    """
    utterances = parse(transcript)
    if not utterances:
        return ""

    pairs = _sentences(utterances)
    stamped = [u.time for u in utterances if u.time]
    questions = [(u, s) for u, s in pairs if s.rstrip().endswith("?")]
    by_speaker = Counter(u.speaker for u in utterances if u.speaker)
    spoken = sum(len(_words(u.text)) for u in utterances)

    lines = ["## Meeting facts", "",
             "*Counted from the transcript, not inferred.*", ""]
    if len(stamped) >= 2:
        lines.append(f"- Ran from **{stamped[0]}** to **{stamped[-1]}**")
    lines.append(f"- {len(utterances)} lines, about {spoken:,} words spoken")

    if len(by_speaker) > 1:
        total = sum(by_speaker.values())
        share = ", ".join(
            f"**{name}** {round(100 * n / total)}% ({n} lines)"
            for name, n in by_speaker.most_common())
        lines.append(f"- Who spoke: {share}")

    if questions:
        lines.append(f"- {len(questions)} question(s) asked:")
        for utt, sentence in questions[:6]:
            lines.append(f"    - {sentence}{_attribute(utt)}")
        if len(questions) > 6:
            lines.append(f"    - …and {len(questions) - 6} more")

    return "\n".join(lines) + "\n"
