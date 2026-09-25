"""The classifier's three-way verdict, and where each verdict is filed."""
import pytest

from core import agent as A
from core.agent import MUSIC, OTHER, VIDEO, TrackVerdict


class TestParseVerdict:
    @pytest.mark.parametrize("raw,expected", [
        ('{"type": "MUSIC", "clean_artist": "A", "clean_title": "T"}', MUSIC),
        ('{"type": "video", "clean_artist": "A", "clean_title": "T"}', VIDEO),
        ('{"type": "Other", "clean_artist": "A", "clean_title": "T"}', OTHER),
        # another word for music, which small models fall back to
        ('{"type": "SONG", "clean_artist": "A", "clean_title": "T"}', MUSIC),
    ])
    def test_types(self, raw, expected):
        assert A.parse_verdict(raw, "a", "t").type == expected

    def test_unknown_type_is_refused(self):
        assert A.parse_verdict('{"type": "PODCAST"}', "a", "t") is None

    def test_json_wrapped_in_prose(self):
        v = A.parse_verdict('Sure! ```{"type": "VIDEO", "clean_title": "X"}```', "Chan", "x")
        assert v == TrackVerdict(VIDEO, "Chan", "X")

    def test_missing_fields_fall_back_to_the_originals(self):
        assert A.parse_verdict('{"type": "MUSIC"}', "Artist", "Title") == TrackVerdict(
            MUSIC, "Artist", "Title")

    def test_non_json(self):
        assert A.parse_verdict("no idea", "a", "t") is None


class TestRoute:
    def test_music_video_goes_to_both_libraries(self):
        assert A.route(TrackVerdict(MUSIC, "a", "t"), has_video=True) == (True, True)

    def test_plain_music_only_to_music(self):
        assert A.route(TrackVerdict(MUSIC, "a", "t"), has_video=False) == (True, False)

    def test_video_only_to_video(self):
        assert A.route(TrackVerdict(VIDEO, "a", "t"), has_video=True) == (False, True)
        # Even if nothing has shown a picture yet: it's a video by verdict.
        assert A.route(TrackVerdict(VIDEO, "a", "t"), has_video=False) == (False, True)

    def test_other_nowhere(self):
        assert A.route(TrackVerdict(OTHER, "a", "t"), has_video=True) == (False, False)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, **kwargs):
        self.prompts.append((kwargs["messages"][0]["content"], kwargs.get("format")))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        class Message:
            content = reply

        class Response:
            message = Message()

        return Response()


@pytest.mark.asyncio
async def test_prompt_carries_site_and_picture():
    client = FakeClient(['{"type": "VIDEO", "clean_artist": "C", "clean_title": "T"}'])
    verdict = await A.CurationAgent(client).classify("C", "T", "youtube.com", True)
    assert verdict.is_video
    prompt = client.prompts[0][0]
    assert 'site="youtube.com"' in prompt and "has_video=yes" in prompt


@pytest.mark.asyncio
async def test_unknown_picture_is_said_so():
    client = FakeClient(['{"type": "MUSIC", "clean_artist": "A", "clean_title": "T"}'])
    await A.CurationAgent(client).classify("A", "T")
    assert "has_video=unknown" in client.prompts[0][0]


@pytest.mark.asyncio
async def test_json_mode_failure_retries_unconstrained():
    client = FakeClient([RuntimeError("500"), '{"type": "MUSIC", "clean_title": "T"}'])
    verdict = await A.CurationAgent(client).classify("A", "T")
    assert verdict.is_music
    assert [fmt for _, fmt in client.prompts] == ["json", None]


class TestStripPromo:
    @pytest.mark.parametrize("raw,clean", [
        ("Levitating (Official Music Video)", "Levitating"),
        ("In the End [Official HD Music Video]", "In the End"),
        ("Numb (Official Video) [4K Remaster]", "Numb"),
        ("Song [Lyrics]", "Song"),
        ("Song (Lyric Video)", "Song"),
        ("Song (Visualizer)", "Song"),
        ("Song (Official Audio) | Artist | Label", "Song"),
        ("Titanium (feat. Sia)", "Titanium (feat. Sia)"),
        ("Strobe (Official Remix)", "Strobe (Official Remix)"),
        ("Creep (Live at Glastonbury)", "Creep (Live at Glastonbury)"),
        ("Tum Hi Ho", "Tum Hi Ho"),
        ("(Official Video)", "(Official Video)"),  # never strip to nothing
    ])
    def test_cases(self, raw, clean):
        assert A.strip_promo(raw) == clean

    def test_applied_to_the_verdict(self):
        v = A.parse_verdict('{"type": "MUSIC", "clean_title": "Levitating (Official Music Video)"}',
                            "Dua Lipa", "x")
        assert v.clean_title == "Levitating"
