import logging
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import play_emotion as play_emotion_module
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import (
    EMOTION_INTENTS,
    PlayEmotion,
    match_dance_intent,
    resolve_emotion_name,
    random_curated_emotion,
    is_dance_emotion_request,
    is_success_emotion_request,
)


AVAILABLE_EMOTIONS = [
    "cheerful1",
    "confused1",
    "no1",
    "no_sad1",
    "no_excited1",
    "resigned1",
    "understanding2",
    "yes_sad1",
]


def test_play_emotion_schema_uses_compact_intents() -> None:
    """Expose compact intents instead of the full recorded-move catalog."""
    emotion_schema = PlayEmotion.parameters_schema["properties"]["emotion"]

    assert emotion_schema["enum"] == list(EMOTION_INTENTS)
    assert "no_sad" in emotion_schema["enum"]
    assert "no_excited" in emotion_schema["enum"]
    assert "no_firm" in emotion_schema["enum"]
    assert "yes_understanding" in emotion_schema["enum"]
    assert "no_confused" not in emotion_schema["enum"]
    assert "oops" not in emotion_schema["enum"]
    assert "yes_sad" not in emotion_schema["enum"]
    assert "yes_proud" not in emotion_schema["enum"]
    assert "loving1" not in emotion_schema["enum"]
    assert "Available emotions" not in emotion_schema["description"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("no_sad1", "no_sad1"),
        ("sad no", "no_sad1"),
        ("no_excited", "no_excited1"),
        ("yes_understanding", "understanding2"),
    ],
)
def test_resolve_emotion_name_accepts_ids_intents_and_yes_no_phrases(requested: str, expected: str) -> None:
    """Resolve exact IDs, compact intents, and exposed yes/no phrase variants."""
    assert resolve_emotion_name(requested, AVAILABLE_EMOTIONS) == expected


def test_is_success_emotion_request() -> None:
    """Only the curated success intent and move IDs count as a success request."""
    assert is_success_emotion_request("success") is True
    assert is_success_emotion_request("success1") is True
    assert is_success_emotion_request("success2") is True
    assert is_success_emotion_request("happy") is False
    assert is_success_emotion_request("random") is False


def test_resolve_emotion_name_maps_success_intent() -> None:
    """The existing emotions library already exposes success1/success2."""
    assert resolve_emotion_name("success", ["success1", "success2", "confused1"]) == "success1"


def test_is_dance_emotion_request() -> None:
    """Only the curated dance intent and move IDs count as a dance request."""
    assert is_dance_emotion_request("dance") is True
    assert is_dance_emotion_request("dance1") is True
    assert is_dance_emotion_request("dance2") is True
    assert is_dance_emotion_request("dance3") is True
    assert is_dance_emotion_request("success") is False
    assert is_dance_emotion_request("random") is False


def test_resolve_emotion_name_maps_dance_intent_to_dance1() -> None:
    """A dance request should play the official dance1 emotion when available."""
    available = ["dance1", "dance2", "dance3", "confused1"]
    assert resolve_emotion_name("dance", available) == "dance1"
    assert resolve_emotion_name("dance1", available) == "dance1"
    assert resolve_emotion_name("dance3", available) == "dance3"


def test_resolve_emotion_name_returns_none_for_random_or_unknown() -> None:
    """Let the caller choose a random fallback when there is no resolved match."""
    assert resolve_emotion_name("random", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("contento", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("totally mysterious mood", AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    "removed_intent",
    [
        "confused no",
        "curious",
        "inquiring",
        "lost",
        "no_confused",
        "oops",
        "proud",
        "uncomfortable",
        "yes proud",
        "yes sad",
        "yes_proud",
        "yes_sad",
    ],
)
def test_resolve_emotion_name_does_not_accept_removed_substitute_intents(removed_intent: str) -> None:
    """Removed intents should not resolve through unrelated substitute moves."""
    assert resolve_emotion_name(removed_intent, AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    ("intent", "poor_options"),
    [
        ("excited", ["success2"]),
        ("grateful", ["helpful1", "loving1"]),
        ("happy", ["loving1"]),
        ("lonely", ["sad1"]),
        ("no", ["no_sad1", "no_excited1"]),
        ("no_excited", ["no1"]),
        ("no_sad", ["downcast1"]),
        ("uncertain", ["resigned1"]),
        ("yes_understanding", ["yes1"]),
    ],
)
def test_resolve_emotion_name_does_not_use_weak_fallbacks(intent: str, poor_options: list[str]) -> None:
    """Do not use loosely related moves when a precise move is unavailable."""
    assert resolve_emotion_name(intent, poor_options) is None


@pytest.mark.parametrize("bad_move", ["cheerful1", "oops1", "oops2", "reprimand3", "understanding1", "yes_sad1"])
def test_resolve_emotion_name_does_not_accept_bad_exact_moves(bad_move: str) -> None:
    """Bad-quality recorded move IDs should not bypass the curated resolver."""
    assert resolve_emotion_name(bad_move, [*AVAILABLE_EMOTIONS, bad_move]) is None


@pytest.mark.parametrize(
    "ambiguous_move",
    [
        "contempt1",
        "curious1",
        "furious1",
        "helpful2",
        "impatient1",
        "incomprehensible2",
        "inquiring1",
        "lost1",
        "proud2",
        "proud3",
        "tired1",
        "uncomfortable1",
        "welcoming1",
    ],
)
def test_resolve_emotion_name_does_not_accept_redundant_ambiguous_exact_moves(ambiguous_move: str) -> None:
    """OK ambiguous moves should be skipped when clear or excellent alternatives exist."""
    assert resolve_emotion_name(ambiguous_move, [*AVAILABLE_EMOTIONS, ambiguous_move]) is None


def test_random_curated_emotion_uses_curated_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random fallback should avoid non-curated moves when curated options exist."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1", "yes_sad1", "confused1"]) == "confused1"
    assert choices_seen == ["confused1"]


def test_random_curated_emotion_falls_back_when_no_curated_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should still return an available move if the curated pool is unavailable."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1"]) == "cheerful1"
    assert choices_seen == ["cheerful1"]


@pytest.mark.asyncio
async def test_play_emotion_queues_resolved_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should queue the resolved recorded-move ID."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())
    result = await tool(deps, emotion="sad no")

    assert result == {"status": "queued", "emotion": "no_sad1"}
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "no_sad1"


@pytest.mark.asyncio
async def test_play_emotion_queues_random_for_unknown_emotion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown explicit values should fall back to a random recorded emotion."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    def fake_choice(emotion_names: list[str]) -> str:
        assert "cheerful1" not in emotion_names
        assert "yes_sad1" not in emotion_names
        return "confused1"

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())
    with caplog.at_level(logging.INFO, logger=play_emotion_module.logger.name):
        result = await tool(deps, emotion="contento")

    assert result == {"status": "queued", "emotion": "confused1"}
    assert "play_emotion: 'contento' did not resolve; using random curated" in caplog.text
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "confused1"


@pytest.mark.asyncio
async def test_play_emotion_allow_random_false_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device-control success must not fall back to a random emotion."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())

    result = await tool(deps, emotion="success", allow_random=False)

    assert result == {"error": "Emotion 'success' is not available"}
    movement_manager.queue_move.assert_not_called()


@pytest.mark.parametrize(
    "transcript",
    [
        "Can you dance?",
        "Can Reachy dance?",
        "Dance for me",
        "Show me some dance moves",
        "Do a dance",
        "Let's dance",
        "Dance please",
        "Show off your dance moves",
        "Can you show me how you dance?",
        "Give me an energetic dance",
        "Dance!",
        "can you dance",
        "CAN YOU DANCE?",
        "hey Reachy, could you show me some dance moves?",
        "reachy dance please",
        "do u dance?",
        "Show me a dance.",
        "Do some dancing.",
        "I want to see you dance.",
        "Give me a dance.",
        "Do your dance.",
        "Can you do some moves?",
        "Show me some moves.",
        "Do an energetic dance.",
        "Give me a crazy dance.",
        "Show me your best dance moves.",
    ],
)
def test_match_dance_intent_triggers_for_requests(transcript: str) -> None:
    """Spoken requests for Reachy to dance should resolve to an official dance emotion."""
    assert match_dance_intent(transcript) in {"dance1", "dance3"}


@pytest.mark.parametrize(
    "transcript",
    [
        "I watched a dance video.",
        "What is dancing?",
        "Tell me about dance.",
        "Do you know the history of dance?",
        "I don't want you to dance.",
        "Do you know what dancing is?",
        "I watched a dance video yesterday.",
    ],
)
def test_match_dance_intent_ignores_non_requests(transcript: str) -> None:
    """Talking about dance without asking Reachy to perform must not queue a dance."""
    assert match_dance_intent(transcript) is None


def test_match_dance_intent_uses_dance1_by_default() -> None:
    """A plain dance request should play dance1, not a random or energetic variant."""
    assert match_dance_intent("Can you dance?") == "dance1"
    assert match_dance_intent("Reachy, can you dance?") == "dance1"


def test_match_dance_intent_uses_dance3_when_energetic() -> None:
    """Explicit energetic or 'best' dance requests may use dance3."""
    assert match_dance_intent("Give me an energetic dance") == "dance3"
    assert match_dance_intent("Do an energetic dance.") == "dance3"
    assert match_dance_intent("Give me a crazy dance.") == "dance3"
    assert match_dance_intent("Show me your best dance moves.") == "dance3"
