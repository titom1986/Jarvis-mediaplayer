import unittest
from unittest.mock import Mock, patch

from tools import radarr


class RadarrRequestTests(unittest.TestCase):
    @patch("tools.radarr.status", return_value={"found": False, "title": "Twelve Monkeys"})
    @patch("tools.radarr.requests.post")
    @patch("tools.radarr.requests.get")
    def test_request_movie_uses_tmdb_and_searches(self, get, post, status):
        lookup = Mock()
        lookup.json.return_value = {"title": "Twelve Monkeys", "tmdbId": 63, "year": 1995}
        lookup.raise_for_status.return_value = None
        roots = Mock()
        roots.json.return_value = [{"id": 1, "path": "/movies"}]
        roots.raise_for_status.return_value = None
        profiles = Mock()
        profiles.json.return_value = [{"id": 7, "name": "HD"}]
        profiles.raise_for_status.return_value = None
        get.side_effect = [lookup, roots, profiles]

        created = Mock()
        created.json.return_value = {"id": 42, "title": "Twelve Monkeys", "tmdbId": 63, "year": 1995, "monitored": True}
        created.raise_for_status.return_value = None
        post.return_value = created

        with patch.dict(radarr.SERVICES, {"radarr": {"url": "http://radarr", "api_key": "secret"}}):
            result = radarr.request_movie(63)

        self.assertTrue(result["added"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tmdbId"], 63)
        self.assertEqual(payload["rootFolderPath"], "/movies")
        self.assertEqual(payload["qualityProfileId"], 7)
        self.assertTrue(payload["monitored"])
        self.assertEqual(payload["addOptions"], {"searchForMovie": True})

    @patch("tools.radarr.status", return_value={"found": True, "title": "Twelve Monkeys", "id": 42})
    @patch("tools.radarr.requests.post")
    @patch("tools.radarr.requests.get")
    def test_request_movie_does_not_duplicate_existing_movie(self, get, post, status):
        lookup = Mock()
        lookup.json.return_value = {"title": "Twelve Monkeys", "tmdbId": 63, "year": 1995}
        lookup.raise_for_status.return_value = None
        roots = Mock()
        roots.json.return_value = [{"id": 1, "path": "/movies"}]
        roots.raise_for_status.return_value = None
        profiles = Mock()
        profiles.json.return_value = [{"id": 7, "name": "HD"}]
        profiles.raise_for_status.return_value = None
        get.side_effect = [lookup, roots, profiles]

        with patch.dict(radarr.SERVICES, {"radarr": {"url": "http://radarr", "api_key": "secret"}}):
            result = radarr.request_movie(63)

        self.assertFalse(result["added"])
        self.assertTrue(result["alreadyExists"])
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
