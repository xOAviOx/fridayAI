"""Tests for SentenceSplitter — the streaming LLM → TTS buffer."""

import pytest

from friday.utils.sentence import SentenceSplitter


class TestPush:
    def test_no_boundary_returns_empty(self):
        s = SentenceSplitter()
        assert s.push("Hello") == []
        assert s.push(", world") == []

    def test_single_sentence_on_period_plus_space(self):
        s = SentenceSplitter(min_length=1)
        # Build up to the boundary across several tokens.
        result = []
        result += s.push("Hello")
        result += s.push(", world.")
        result += s.push(" ")
        assert result == ["Hello, world."]

    def test_exclamation_and_question_also_split(self):
        s = SentenceSplitter(min_length=1)
        r = []
        r += s.push("Stop! ")
        r += s.push("Why? ")
        assert r == ["Stop!", "Why?"]

    def test_min_length_suppresses_short_candidates(self):
        # "Dr." is only 3 chars — must not fire as its own sentence.
        # The splitter advances past it; the next sentence starts from
        # whatever follows.
        s = SentenceSplitter(min_length=10)
        r = []
        r += s.push("Dr. ")           # "Dr." is too short → skipped
        r += s.push("Smith is here.") # Buffer: "Smith is here." — no trailing space yet
        r += s.push(" Done.")         # Now sees "Smith is here. Done." → fires
        # "Dr." must NOT appear as its own sentence.
        assert "Dr." not in r
        # The sentence that *does* fire contains "Smith is here."
        assert any("Smith is here." in s for s in r)

    def test_multiple_sentences_in_one_push(self):
        s = SentenceSplitter(min_length=1)
        r = s.push("First. Second. Third. ")
        assert r == ["First.", "Second.", "Third."]

    def test_empty_token_noop(self):
        s = SentenceSplitter()
        assert s.push("") == []

    def test_ellipsis_acts_as_boundary(self):
        s = SentenceSplitter(min_length=1)
        r = []
        r += s.push("Hmm… ")
        r += s.push("OK")
        assert "Hmm…" in r


class TestFlush:
    def test_flush_returns_remaining_buffer(self):
        s = SentenceSplitter()
        s.push("No period here")
        result = s.flush()
        assert result == ["No period here"]

    def test_flush_empty_buffer_returns_empty(self):
        s = SentenceSplitter()
        assert s.flush() == []

    def test_flush_clears_buffer(self):
        s = SentenceSplitter()
        s.push("some text")
        s.flush()
        assert s.flush() == []

    def test_flush_returns_only_trailing_fragment(self):
        s = SentenceSplitter(min_length=1)
        r = []
        r += s.push("Complete sentence. ")
        r += s.push("Incomplete")
        tail = s.flush()
        assert r == ["Complete sentence."]
        assert tail == ["Incomplete"]


class TestReset:
    def test_reset_clears_buffer(self):
        s = SentenceSplitter()
        s.push("partial")
        s.reset()
        assert s.flush() == []

    def test_reset_allows_reuse(self):
        s = SentenceSplitter(min_length=1)
        s.push("First sentence. ")
        s.flush()
        s.reset()
        r = s.push("Second. ")
        assert r == ["Second."]


class TestEndToEnd:
    def test_full_streaming_scenario(self):
        """Simulate a streaming LLM response token by token."""
        tokens = [
            "I ", "opened ", "Spotify ", "and ", "started ",
            "playing ", "music. ", "Enjoy ", "your ", "tunes!",
        ]
        s = SentenceSplitter(min_length=5)
        sentences = []
        for tok in tokens:
            sentences.extend(s.push(tok))
        sentences.extend(s.flush())

        assert len(sentences) == 2
        assert sentences[0] == "I opened Spotify and started playing music."
        assert sentences[1] == "Enjoy your tunes!"

    def test_single_word_reply_via_flush(self):
        """A one-word reply with no punctuation comes out of flush."""
        s = SentenceSplitter(min_length=1)
        result = s.push("Sure")
        result += s.flush()
        assert result == ["Sure"]
