import unittest
from unittest.mock import Mock, patch

from tools import plex, radarr, seerr, sonarr


def response(data=None, status=200):
    r = Mock()
    r.status_code = status
    r.ok = 200 <= status < 400
    r.json.return_value = {} if data is None else data
    r.text = ""
    r.raise_for_status.return_value = None
    return r


class HttpContractTests(unittest.TestCase):
    def test_seerr_free_text_searches_use_rfc3986_urls(self):
        with patch.object(seerr, "SEERR_API_KEY", "secret"), patch("tools.seerr.requests.get", return_value=response({"results": []})) as get:
            seerr.search("Ocean's Eleven & More?")
            url = get.call_args.args[0]
            self.assertIn("query=Ocean%27s%20Eleven%20%26%20More%3F", url)
            self.assertNotIn("params", get.call_args.kwargs)

            seerr.search_keyword("time travel / paradox")
            url = get.call_args.args[0]
            self.assertIn("query=time%20travel%20%2F%20paradox", url)
            self.assertIn("page=1", url)
            self.assertNotIn("params", get.call_args.kwargs)

    def test_seerr_structured_discover_keeps_typed_query_params(self):
        with patch.object(seerr, "SEERR_API_KEY", "secret"), patch("tools.seerr.requests.get", return_value=response({"results": []})) as get:
            seerr.discover("movie", page=2, genre_ids=[878], keyword_ids=[4563], date_from="1995-01-01", date_to="2015-12-31")
            params = get.call_args.kwargs["params"]
            self.assertEqual(params["page"], 2)
            self.assertEqual(params["genre"], "878")
            self.assertEqual(params["keywords"], "4563")
            self.assertEqual(params["primaryReleaseDateGte"], "1995-01-01")
            self.assertEqual(params["primaryReleaseDateLte"], "2015-12-31")

    def test_plex_free_text_stays_in_requests_params_and_token_never_enters_path(self):
        payload = {"MediaContainer": {"Hub": []}}
        with patch.object(plex, "PLEX_TOKEN", "secret"), patch("tools.plex.requests.get", return_value=response(payload)) as get:
            plex.status("Ocean's Eleven & More?")
            self.assertEqual(get.call_args.args[0], f"{plex.PLEX_URL}/hubs/search")
            self.assertEqual(get.call_args.kwargs["params"]["query"], "Ocean's Eleven & More?")
            self.assertEqual(get.call_args.kwargs["params"]["X-Plex-Token"], "secret")

    def test_radarr_http_contract_uses_numeric_query_and_json_body(self):
        cfg = {"url": "http://radarr", "api_key": "secret", "root_folder": "/movies", "default_profile": "HD", "french_profile": "FR"}
        with patch.dict(radarr.SERVICES, {"radarr": cfg}), patch("tools.radarr.status", return_value={"found": False}), patch("tools.radarr.requests.get") as get, patch("tools.radarr.requests.post") as post:
            get.side_effect = [
                response({"title": "Ocean's Eleven & More?", "tmdbId": 161}),
                response([{"id": 1, "path": "/movies"}]),
                response([{"id": 2, "name": "HD"}]),
            ]
            post.return_value = response({"id": 9, "tmdbId": 161, "title": "Ocean's Eleven & More?", "year": 2001, "monitored": True})
            radarr.request_movie(161)
            self.assertEqual(get.call_args_list[0].kwargs["params"], {"tmdbId": 161})
            self.assertEqual(post.call_args.kwargs["json"]["title"], "Ocean's Eleven & More?")
            self.assertNotIn("Ocean", post.call_args.args[0])

    def test_sonarr_queue_uses_validated_queue_contract(self):
        cfg = {"url": "http://sonarr", "api_key": "secret"}
        series = [{"id": 7, "title": "Show"}]
        queue = {"records": [{
            "id": 11, "seriesId": 7, "episodeId": 12, "title": "Episode",
            "status": "downloading", "size": 1000, "sizeleft": 250,
            "estimatedCompletionTime": "2026-09-25T12:30:00Z",
            "quality": {"quality": {"name": "WEBDL-1080p"}},
            "languages": [{"name": "French"}], "downloadClient": "Deluge"
        }]}
        with patch.dict(sonarr.SERVICES, {"sonarr": cfg}), patch("tools.sonarr.requests.get") as get:
            get.side_effect = [response(series), response(queue)]
            result = sonarr.queue_status("Show")
            self.assertTrue(result["inQueue"])
            self.assertEqual(result["items"][0]["sizeleft"], 250)
            self.assertEqual(get.call_args_list[1].kwargs["params"], {"page": 1, "pageSize": 100})

    def test_sonarr_title_is_filtered_locally_not_put_in_url(self):
        with patch.dict(sonarr.SERVICES, {"sonarr": {"url": "http://sonarr", "api_key": "secret"}}), patch("tools.sonarr.requests.get", return_value=response([])) as get:
            sonarr.status("Marvel's Agents & More?")
            self.assertEqual(get.call_args.args[0], "http://sonarr/api/v3/series")
            self.assertNotIn("params", get.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
