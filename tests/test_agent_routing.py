import unittest
from unittest.mock import patch

import agent


class AgentRoutingTests(unittest.TestCase):
    def setUp(self):
        agent.catalog_sets.reset()

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

    @patch("agent.catalog_sets.execute_constraint_group")
    def test_parallel_constraint_batch_and_or_exclude(self, execute_group):
        include0 = agent.catalog_sets._store({2, 3, 4}, "movie", "include0")
        include1 = agent.catalog_sets._store({9}, "movie", "include1")
        banned = agent.catalog_sets._store({3, 4, 9}, "movie", "banned")
        execute_group.side_effect = [include0, include1, banned]

        result = agent._compose_catalog_batch([
            ("catalog_person", {"name": "A", "media_type": "movie", "group": 0}),
            ("catalog_genre", {"name": "Action", "media_type": "movie", "group": 1}),
            ("catalog_years", {"year_from": 1998, "year_to": 1998, "media_type": "movie", "exclude": True}),
        ])
        self.assertEqual(agent.catalog_sets._get(result["set"])["ids"], {2})
        # The exclusion group must refine the already-composed include subset.
        exclusion_call = execute_group.call_args_list[2]
        self.assertIsNotNone(exclusion_call.kwargs.get("source"))

    @patch("tools.catalog_sets.materialize_constraint")
    @patch("tools.catalog_sets.estimate_constraint")
    def test_exclusion_refines_existing_subset_without_broad_materialization(self, estimate, materialize):
        source = agent.catalog_sets._store({10, 11, 12}, "movie", "included")
        estimate.return_value = {"count": 50000}
        filtered = agent.catalog_sets._store({11}, "movie", "excluded")
        materialize.return_value = filtered

        result = agent.catalog_sets.execute_constraint_group([
            ("catalog_years", {
                "year_from": 1998, "year_to": 1998, "media_type": "movie"
            })
        ], source=source["set"])

        self.assertEqual(result["set"], filtered["set"])
        materialize.assert_called_once()
        self.assertEqual(materialize.call_args.kwargs["source"], source["set"])
        self.assertNotIn("seed_ids", materialize.call_args.kwargs)

    @patch("tools.catalog_sets.materialize_constraint")
    @patch("tools.catalog_sets.estimate_constraint")
    def test_best_seed_is_selected_by_count_not_semantic_type(self, estimate, materialize):
        # year is cheapest here; another fixture can make any other type win.
        estimate.side_effect = [
            {"count": 5000}, {"count": 12}, {"count": 300}
        ]
        seed = agent.catalog_sets._store(set(range(12)), "movie", "year-seed")
        after_person = agent.catalog_sets._store({1, 2, 3}, "movie", "person-refined")
        after_genre = agent.catalog_sets._store({2, 3}, "movie", "genre-refined")
        materialize.side_effect = [seed, after_genre, after_person]

        result = agent.catalog_sets.execute_constraint_group([
            ("catalog_person", {"name": "A", "media_type": "movie"}),
            ("catalog_years", {"year_from": 1990, "year_to": 1999, "media_type": "movie"}),
            ("catalog_genre", {"name": "Science Fiction", "media_type": "movie"}),
        ])
        first = materialize.call_args_list[0]
        self.assertEqual(first.args[0], "catalog_years")
        self.assertEqual(result["set"], after_person["set"])

    @patch("tools.catalog_sets.seerr.discover")
    @patch("tools.catalog_sets.media_search._resolve_genre_ids", return_value=[878])
    @patch("tools.catalog_sets.media_search._resolve_keyword_ids", return_value=[])
    def test_genre_estimate_reads_only_first_discover_page(self, kw, genre, discover):
        discover.return_value = {
            "page": 1, "totalPages": 100, "totalResults": 1987,
            "results": [{"id": i} for i in range(20)]
        }
        result = agent.catalog_sets.estimate_constraint(
            "catalog_genre", {"name": "Science Fiction", "media_type": "movie"}
        )
        self.assertEqual(result["count"], 1987)
        self.assertEqual(discover.call_count, 1)

    def test_unknown_tool_is_nonfatal(self):
        self.assertIn("error", agent.execute_tool("does_not_exist", {}))


if __name__ == "__main__":
    unittest.main()
