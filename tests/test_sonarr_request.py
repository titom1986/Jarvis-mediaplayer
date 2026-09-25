import unittest
from unittest.mock import Mock, patch

from tools import sonarr


CFG = {
    "url": "http://sonarr",
    "api_key": "secret",
    "root_folder": "/home/titom/data/series",
    "default_profile": "HD - 720p/1080p",
    "french_profile": "French HD - 720p/1080p",
}


def response(data):
    item = Mock()
    item.json.return_value = data
    item.raise_for_status.return_value = None
    return item


class SonarrRequestTests(unittest.TestCase):
    @patch("tools.sonarr.requests.post")
    @patch("tools.sonarr.requests.get")
    def test_add_series_then_search_whole_series(self, get, post):
        get.side_effect = [
            response([]),
            response([{"title": "Show", "tvdbId": 100}]),
            response([{"id": 2, "path": "/home/titom/data/series"}]),
            response([{"id": 6, "name": "HD - 720p/1080p"}]),
        ]
        post.side_effect = [
            response({"id": 42, "title": "Show", "tvdbId": 100}),
            response({"id": 900}),
        ]
        with patch.dict(sonarr.SERVICES, {"sonarr": CFG}):
            result = sonarr.request_series(100, scope="series")
        self.assertTrue(result["requested"])
        self.assertTrue(result["added"])
        self.assertEqual(post.call_args_list[1].kwargs["json"], {"name": "SeriesSearch", "seriesId": 42})

    @patch("tools.sonarr.requests.post")
    @patch("tools.sonarr.requests.get")
    def test_existing_series_searches_one_season(self, get, post):
        get.return_value = response([{"id": 42, "title": "Show", "tvdbId": 100}])
        post.return_value = response({"id": 901})
        with patch.dict(sonarr.SERVICES, {"sonarr": CFG}):
            result = sonarr.request_series(100, scope="season", season=3)
        self.assertEqual(post.call_args.kwargs["json"], {"name": "SeasonSearch", "seriesId": 42, "seasonNumber": 3})
        self.assertEqual(result["season"], 3)

    @patch("tools.sonarr.requests.post")
    @patch("tools.sonarr.requests.get")
    def test_episode_request_resolves_native_episode_id(self, get, post):
        get.side_effect = [
            response([{"id": 42, "title": "Show", "tvdbId": 100}]),
            response([
                {"id": 501, "seasonNumber": 3, "episodeNumber": 6, "hasFile": False},
                {"id": 502, "seasonNumber": 3, "episodeNumber": 7, "hasFile": False},
            ]),
        ]
        post.return_value = response({"id": 902})
        with patch.dict(sonarr.SERVICES, {"sonarr": CFG}):
            result = sonarr.request_series(100, scope="episode", season=3, episode=7)
        self.assertEqual(post.call_args.kwargs["json"], {"name": "EpisodeSearch", "episodeIds": [502]})
        self.assertTrue(result["requested"])

    @patch("tools.sonarr.requests.post")
    @patch("tools.sonarr.requests.get")
    def test_rerequest_episode_searches_even_when_file_exists(self, get, post):
        get.side_effect = [
            response([{"id": 42, "title": "Show", "tvdbId": 100}]),
            response([{"id": 502, "seasonNumber": 3, "episodeNumber": 7, "hasFile": True}]),
        ]
        post.return_value = response({"id": 903})
        with patch.dict(sonarr.SERVICES, {"sonarr": CFG}):
            result = sonarr.request_series(100, scope="episode", season=3, episode=7, force=True)
        self.assertEqual(post.call_args.kwargs["json"], {"name": "EpisodeSearch", "episodeIds": [502]})
        self.assertTrue(result["force"])


if __name__ == "__main__":
    unittest.main()
