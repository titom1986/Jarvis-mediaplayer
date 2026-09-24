import unittest
from unittest.mock import patch

import agent


class AgentRoutingTests(unittest.TestCase):
    def test_catalog_refinements_forward_source(self):
        with patch("agent.catalog_sets.genre", return_value={"set": "s2"}) as genre:
            agent.execute_tool("catalog_genre", {
                "name": "Science Fiction", "media_type": "movie", "source": "s1"
            })
            genre.assert_called_once_with("Science Fiction", "movie", "s1")

        with patch("agent.catalog_sets.keyword", return_value={"set": "s3"}) as keyword:
            agent.execute_tool("catalog_keyword", {
                "name": "time travel", "media_type": "movie", "source": "s2"
            })
            keyword.assert_called_once_with("time travel", "movie", "s2")

        with patch("agent.catalog_sets.years", return_value={"set": "s4"}) as years:
            agent.execute_tool("catalog_years", {
                "year_from": 1990, "year_to": 1999,
                "media_type": "movie", "source": "s3"
            })
            years.assert_called_once_with(1990, 1999, "movie", "s3")

    def test_catalog_set_algebra_routing(self):
        with patch("agent.catalog_sets.combine", return_value={"set": "s3"}) as combine:
            agent.execute_tool("catalog_combine", {
                "operation": "union", "sets": ["s1", "s2"]
            })
            combine.assert_called_once_with("union", ["s1", "s2"])

        with patch("agent.catalog_sets.subtract", return_value={"set": "s5"}) as subtract:
            agent.execute_tool("catalog_subtract", {"source": "s3", "remove": "s4"})
            subtract.assert_called_once_with("s3", "s4")

        with patch("agent.catalog_sets.results", return_value={"results": []}) as results:
            agent.execute_tool("catalog_results", {"set": "s5", "limit": 5})
            results.assert_called_once_with("s5", 5)

    def test_plex_radarr_sonarr_routing(self):
        with patch("agent.plex.status", return_value={"found": False}) as plex:
            agent.execute_tool("plex_status", {"title": "Movie"})
            plex.assert_called_once_with("Movie")

        with patch("agent.radarr.status", return_value={"found": False}) as radarr:
            agent.execute_tool("radarr_status", {"title": "Movie"})
            radarr.assert_called_once_with("Movie")

        with patch("agent.radarr.queue_status", return_value={"found": False}) as queue:
            agent.execute_tool("radarr_queue_status", {"title": "Movie"})
            queue.assert_called_once_with("Movie")

        with patch("agent.radarr.request_movie", return_value={"added": True}) as request:
            agent.execute_tool("radarr_request_movie", {"tmdb_id": 123, "french": True})
            request.assert_called_once_with(123, french=True)

        with patch("agent.sonarr.status", return_value={"found": False}) as sonarr:
            agent.execute_tool("sonarr_status", {"title": "Series"})
            sonarr.assert_called_once_with("Series")

    def test_unknown_tool_is_nonfatal(self):
        self.assertIn("error", agent.execute_tool("does_not_exist", {}))


if __name__ == "__main__":
    unittest.main()
