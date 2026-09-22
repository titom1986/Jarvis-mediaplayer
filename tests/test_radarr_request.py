import unittest
from unittest.mock import Mock, patch

from tools import radarr


CFG = {
    "url": "http://radarr",
    "api_key": "secret",
    "root_folder": "/home/titom/data/movies",
    "default_profile": "HD - 720p/1080p",
    "french_profile": "French 720p/1080p",
}


def response(data):
    item = Mock()
    item.json.return_value = data
    item.raise_for_status.return_value = None
    return item


class RadarrRequestTests(unittest.TestCase):
    @patch("tools.radarr.status", return_value={"found": False, "title": "Twelve Monkeys"})
    @patch("tools.radarr.requests.post")
    @patch("tools.radarr.requests.get")
    def test_request_movie_uses_configured_default_profile(self, get, post, status):
        get.side_effect = [
            response({"title": "Twelve Monkeys", "tmdbId": 63, "year": 1995}),
            response([
                {"id": 1, "path": "/home/titom/Videos/Movies"},
                {"id": 2, "path": "/home/titom/data/movies"},
            ]),
            response([
                {"id": 1, "name": "Any"},
                {"id": 6, "name": "HD - 720p/1080p"},
                {"id": 7, "name": "French 720p/1080p"},
            ]),
        ]
        post.return_value = response(
            {"id": 42, "title": "Twelve Monkeys", "tmdbId": 63, "year": 1995, "monitored": True}
        )

        with patch.dict(radarr.SERVICES, {"radarr": CFG}):
            result = radarr.request_movie(63)

        self.assertTrue(result["added"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["rootFolderPath"], "/home/titom/data/movies")
        self.assertEqual(payload["qualityProfileId"], 6)
        self.assertEqual(result["qualityProfile"], "HD - 720p/1080p")

    @patch("tools.radarr.status", return_value={"found": False, "title": "Twelve Monkeys"})
    @patch("tools.radarr.requests.post")
    @patch("tools.radarr.requests.get")
    def test_request_movie_uses_french_profile_only_when_requested(self, get, post, status):
        get.side_effect = [
            response({"title": "Twelve Monkeys", "tmdbId": 63, "year": 1995}),
            response([{"id": 2, "path": "/home/titom/data/movies"}]),
            response([
                {"id": 6, "name": "HD - 720p/1080p"},
                {"id": 7, "name": "French 720p/1080p"},
            ]),
        ]
        post.return_value = response(
            {"id": 42, "title": "Twelve Monkeys", "tmdbId": 63, "year": 1995, "monitored": True}
        )

        with patch.dict(radarr.SERVICES, {"radarr": CFG}):
            result = radarr.request_movie(63, french=True)

        self.assertTrue(result["added"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["qualityProfileId"], 7)
        self.assertEqual(result["qualityProfile"], "French 720p/1080p")

    @patch("tools.radarr.status", return_value={"found": True, "title": "Twelve Monkeys", "id": 42})
    @patch("tools.radarr.requests.post")
    @patch("tools.radarr.requests.get")
    def test_request_movie_does_not_duplicate_existing_movie(self, get, post, status):
        get.return_value = response({"title": "Twelve Monkeys", "tmdbId": 63, "year": 1995})

        with patch.dict(radarr.SERVICES, {"radarr": CFG}):
            result = radarr.request_movie(63)

        self.assertFalse(result["added"])
        self.assertTrue(result["alreadyExists"])
        post.assert_not_called()
        self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
