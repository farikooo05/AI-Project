"""Build a high-precision additional emotion dataset for later transformer retraining."""

from __future__ import annotations

from itertools import product
import random

import pandas as pd

from emotion_detector.data_loader import EXPECTED_LABELS


STRICT_LABEL_GUIDELINES: dict[str, dict[str, list[str]]] = {
    "joy": {
        "include": [
            "clear happiness, delight, cheerfulness, or relief with obvious positive emotion",
            "statements where the speaker directly sounds happy, pleased, excited, or uplifted",
            "positive reactions that feel warm, bright, celebratory, or emotionally light",
        ],
        "exclude": [
            "gratitude without visible happiness",
            "admiration, approval, or love without explicit joy",
            "mixed surprise and joy where shock is stronger than happiness",
        ],
    },
    "sadness": {
        "include": [
            "grief, disappointment, emotional pain, loneliness, hopelessness, or feeling low",
            "comments that clearly sound hurt, broken, heavy, empty, or down",
        ],
        "exclude": [
            "blaming or attacking language that reads more like anger",
            "flat reporting of bad events without emotional pain",
        ],
    },
    "anger": {
        "include": [
            "frustration, hostility, outrage, resentment, irritation, or feeling furious",
            "comments with clear emotional heat, hostility, or fed-up energy",
        ],
        "exclude": [
            "cold criticism without emotional intensity",
            "sadness phrasing focused on pain rather than hostility",
        ],
    },
    "fear": {
        "include": [
            "anxiety, dread, worry, panic, feeling unsafe, or being scared",
            "comments where the speaker clearly anticipates danger, harm, or loss",
        ],
        "exclude": [
            "generic suspense or horror references without personal fear",
            "surprise reactions where shock is stronger than fear",
        ],
    },
    "surprise": {
        "include": [
            "shock, amazement, disbelief, and strongly unexpected reactions",
            "comments where the speaker clearly did not expect what just happened",
        ],
        "exclude": [
            "general excitement without an unexpected event",
            "fearful shock where dread is stronger than surprise",
        ],
    },
    "neutral": {
        "include": [
            "emotionally flat, factual, descriptive, or procedural statements",
            "simple updates, observations, reminders, or ordinary comments with low affect",
        ],
        "exclude": [
            "sarcasm, profanity, aggression, enthusiasm, or disappointment",
            "language with clear emotional tone even if mild",
        ],
    },
}


CURATED_SOURCE = "curated_additional"
CURATED_QUALITY_TAG = "high_precision"


def _sentence_case(text: str) -> str:
    """Capitalize the first visible character while preserving punctuation."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _build_sentences(
    openers: list[str],
    middles: list[str],
    endings: list[str],
    templates: list[str],
) -> list[str]:
    """Build diverse candidate sentences from phrase pools and templates."""
    sentences: list[str] = []
    for template, opener, middle, ending in product(templates, openers, middles, endings):
        text = template.format(
            opener=opener,
            middle=middle,
            ending=ending,
            opener_cap=_sentence_case(opener),
            middle_cap=_sentence_case(middle),
        )
        text = " ".join(text.split())
        if text not in sentences:
            sentences.append(text)
    return sentences


def _joy_candidates() -> list[str]:
    openers = [
        "this honestly",
        "well, that",
        "not gonna lie, that",
        "okay, that",
        "today just",
        "finally, something",
        "for once, something",
        "the whole thing",
        "that little moment",
        "somehow this",
        "wow, that",
        "I cannot even explain it, but this",
    ]
    middles = [
        "made my day",
        "went so much better than I expected",
        "put me in such a good mood",
        "felt like a real win",
        "actually turned everything around",
        "left me smiling for hours",
        "felt genuinely wonderful",
        "was exactly what I needed",
        "hit me with pure happiness",
        "felt so bright and easy",
        "worked out perfectly",
        "turned out beautifully",
    ]
    endings = [
        "seriously.",
        "for real.",
        "and I am still smiling.",
        "and I love that.",
        "and I cannot stop grinning.",
        "in the best way.",
        "and it feels amazing.",
        "and I needed that so much.",
        "and I am so happy right now.",
        "and it lifted my whole mood.",
    ]
    templates = [
        "{opener_cap} {middle} {ending}",
        "{middle_cap} {ending}",
        "{opener_cap} {middle}.",
        "{opener_cap} {middle}, {ending}",
        "{middle_cap}, honestly.",
        "{opener_cap}? {middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "This made my day.",
        "I cannot believe how well that went.",
        "Everything clicked at once and I am so happy about it.",
        "That news put me in the best mood.",
        "I am actually so excited right now.",
        "This turned out even better than I hoped.",
        "Best part of my week, easily.",
        "I have not smiled like this in a while.",
        "That felt amazing, honestly.",
        "I am still riding that happy feeling.",
    ]
    return base + extras


def _sadness_candidates() -> list[str]:
    openers = [
        "that",
        "honestly, this",
        "right now this",
        "tonight this",
        "everything about that",
        "the whole situation",
        "that message",
        "this silence",
        "that memory",
        "it all",
        "this really",
        "the way this ended",
    ]
    middles = [
        "hurts more than I expected",
        "left me feeling empty",
        "is genuinely upsetting",
        "broke my heart a little",
        "has me feeling really low",
        "made everything feel heavy",
        "hit me in a bad way",
        "still hurts to think about",
        "feels painfully final",
        "just made me want to cry",
        "drained me emotionally",
        "landed harder than it should have",
    ]
    endings = [
        "tonight.",
        "right now.",
        "if I am honest.",
        "and I do not know what to do with that.",
        "and it still hurts.",
        "and I cannot shake it.",
        "and it feels awful.",
        "and now the whole day feels heavy.",
        "and I just feel sad.",
        "and it is sitting with me.",
    ]
    templates = [
        "{opener_cap} {middle} {ending}",
        "{middle_cap} {ending}",
        "{opener_cap} {middle}.",
        "{opener_cap} {middle}, {ending}",
        "{middle_cap}, honestly.",
        "{opener_cap}? {middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "That is honestly upsetting.",
        "I did not think it would hurt this much.",
        "This feels heavier than I can explain.",
        "I just feel really down tonight.",
        "That left me completely heartbroken.",
        "Nothing about this feels okay.",
        "I am trying to act normal, but this hurts.",
        "That one really got to me.",
        "I keep thinking about it and it just makes me sad.",
        "This hurts in a quiet way.",
    ]
    return base + extras


def _anger_candidates() -> list[str]:
    openers = [
        "seriously, this",
        "that",
        "honestly, this",
        "the way they handled that",
        "this whole mess",
        "that comment",
        "the fact that this happened again",
        "all of this",
        "this nonsense",
        "that decision",
        "this kind of behavior",
        "the entire situation",
    ]
    middles = [
        "is making me furious",
        "is unbelievably irritating",
        "just made me snap",
        "is so disrespectful",
        "has me absolutely mad",
        "is infuriating",
        "pushed me over the edge",
        "is getting on my last nerve",
        "makes my blood boil",
        "is unbelievably rude",
        "is beyond frustrating",
        "has me ready to walk away",
    ]
    endings = [
        "right now.",
        "for real.",
        "and I am done with it.",
        "and I am not staying quiet about it.",
        "and I hate this.",
        "and it is making me so angry.",
        "and I have had enough.",
        "and I cannot just ignore that.",
        "and I want them to stop immediately.",
        "and it is not okay.",
    ]
    templates = [
        "{opener_cap} {middle} {ending}",
        "{middle_cap} {ending}",
        "{opener_cap} {middle}.",
        "{opener_cap} {middle}, {ending}",
        "{middle_cap}, honestly.",
        "{opener_cap}? {middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "Leave me alone.",
        "I am so tired of this.",
        "That was completely out of line.",
        "This is making me really angry.",
        "No, that is not okay.",
        "I am done being patient about this.",
        "That crossed a line.",
        "I am actually furious right now.",
        "This is beyond annoying.",
        "I cannot believe they thought that was acceptable.",
    ]
    return base + extras


def _fear_candidates() -> list[str]:
    openers = [
        "that",
        "right now I",
        "honestly I",
        "this",
        "the way that happened",
        "all of this",
        "that sound",
        "the thought of it",
        "this whole situation",
        "what just happened",
        "that warning",
        "the possibility of that",
    ]
    middles = [
        "really scared me",
        "has me feeling unsafe",
        "made my stomach drop",
        "is honestly terrifying",
        "has me genuinely worried",
        "made me panic for a second",
        "has me on edge",
        "feels dangerous",
        "is making me anxious",
        "left me shaken",
        "made it hard to breathe for a moment",
        "has me fearing the worst",
    ]
    endings = [
        "not even kidding.",
        "right now.",
        "and I cannot calm down.",
        "and I am still shaky.",
        "and I hate this feeling.",
        "and now I am really worried.",
        "and it feels bad.",
        "and I do not feel safe.",
        "and my chest is tight.",
        "and my mind went straight to the worst outcome.",
    ]
    templates = [
        "{opener_cap} {middle} {ending}",
        "{middle_cap} {ending}",
        "{opener_cap} {middle}.",
        "{opener_cap} {middle}, {ending}",
        "{middle_cap}, honestly.",
        "{opener_cap}? {middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "That scared me.",
        "I am honestly afraid right now.",
        "This has me really worried.",
        "I do not like how unsafe this feels.",
        "My heart dropped when I saw that.",
        "I am trying to stay calm, but I am scared.",
        "That genuinely frightened me.",
        "I cannot stop thinking something bad is about to happen.",
        "This is making me anxious in a real way.",
        "Nope, that scared me.",
    ]
    return base + extras


def _surprise_candidates() -> list[str]:
    openers = [
        "wait",
        "wow",
        "okay",
        "honestly",
        "well",
        "seriously",
        "that",
        "this",
        "the fact that this happened",
        "what just happened",
        "all of that",
        "the ending",
    ]
    middles = [
        "I did not see that coming",
        "completely caught me off guard",
        "was so unexpected",
        "actually shocked me",
        "left me staring at the screen",
        "threw me off completely",
        "was not what I expected at all",
        "made me do a double take",
        "was a real surprise",
        "felt like it came out of nowhere",
        "was genuinely startling",
        "just blindsided me",
    ]
    endings = [
        "at all.",
        "for real.",
        "and now I am just staring.",
        "and I need a second.",
        "and I am still processing it.",
        "and I honestly do not know what to say.",
        "and that was wild.",
        "and I was not ready for that.",
        "and that threw me.",
        "and I just blinked at my phone.",
    ]
    templates = [
        "{opener_cap}, {middle} {ending}",
        "{middle_cap} {ending}",
        "{opener_cap}... {middle}.",
        "{opener_cap} {middle}.",
        "{middle_cap}, honestly.",
        "{opener_cap}? {middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "Wait... what just happened?",
        "I genuinely did not expect that.",
        "That came out of nowhere.",
        "I am honestly shocked.",
        "Well, that was unexpected.",
        "I had to read that twice.",
        "That completely caught me off guard.",
        "Did not see that one coming.",
        "Okay, wow.",
        "That was a real surprise.",
    ]
    return base + extras


def _neutral_candidates() -> list[str]:
    openers = [
        "the meeting",
        "the update",
        "the package",
        "the train",
        "the report",
        "the office",
        "the event",
        "the form",
        "the ticket",
        "the file",
        "the class",
        "the delivery",
    ]
    middles = [
        "starts at nine",
        "is scheduled for tomorrow",
        "was uploaded to the drive",
        "arrives at noon",
        "was sent this morning",
        "opens at ten",
        "is still on the calendar",
        "is available online",
        "was confirmed earlier today",
        "is in the shared folder",
        "begins in room twelve",
        "is expected by Friday",
    ]
    endings = [
        ".",
        " if you need to check it.",
        " according to the schedule.",
        " for the current plan.",
        " and the details are in the note.",
        " based on the latest update.",
        " as listed in the document.",
        " for anyone following the timeline.",
        " with no other changes.",
        " and that is the current status.",
    ]
    templates = [
        "{opener_cap} {middle}{ending}",
        "{middle_cap}{ending}",
        "{opener_cap} {middle}.",
        "{opener_cap}: {middle}.",
        "{middle_cap}.",
    ]
    base = _build_sentences(openers, middles, endings, templates)
    extras = [
        "The meeting starts at nine.",
        "The file is in the shared folder.",
        "The package arrived this morning.",
        "The event is still scheduled for Friday.",
        "The report was sent earlier.",
        "The train leaves from platform three.",
        "The office opens at ten.",
        "The form is available online.",
        "The update was posted this afternoon.",
        "The document has the final version.",
    ]
    return base + extras


def _build_candidates_for_label(label: str) -> list[str]:
    """Generate semantically strong candidate sentences for one label."""
    builders = {
        "joy": _joy_candidates,
        "sadness": _sadness_candidates,
        "anger": _anger_candidates,
        "fear": _fear_candidates,
        "surprise": _surprise_candidates,
        "neutral": _neutral_candidates,
    }
    if label not in builders:
        raise ValueError(f"Unsupported label: {label}")
    return builders[label]()


def build_curated_examples(
    per_class: int = 240,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a balanced additional dataset with semantically strong single-label examples.

    The generator is deterministic and intentionally conservative. It produces
    more candidates than needed, then the quality-control stage removes
    duplicates and near-duplicates before the final per-class sample is taken.
    """
    random_generator = random.Random(random_seed)
    rows: list[dict[str, str]] = []

    for label in EXPECTED_LABELS:
        candidates = _build_candidates_for_label(label)
        random_generator.shuffle(candidates)
        for text in candidates[: per_class * 5]:
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": CURATED_SOURCE,
                    "quality_tag": CURATED_QUALITY_TAG,
                }
            )

    return pd.DataFrame(rows)
