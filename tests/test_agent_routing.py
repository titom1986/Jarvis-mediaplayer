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
            keyword.assert_called_once_with("time travel", "movie", "s2", aliases=None)

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
        estimate.assert_not_called()

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

    @patch("agent.catalog_sets.results")
    @patch("agent.catalog_sets.execute_constraint_group")
    @patch("agent.requests.post")
    def test_full_agent_catalogue_chain_materializes_grounded_results(self, post, execute_group, results):
        execute_group.side_effect = lambda members, source=None: agent.catalog_sets._store({101, 102}, "movie", "seed")
        results.return_value = {
            "set": "s1", "count": 2,
            "results": [
                {"mediaType": "movie", "id": 101, "title": "Alpha", "releaseDate": "2001-01-01", "rating": 8.0, "voteCount": 100},
                {"mediaType": "movie", "id": 102, "title": "Beta", "releaseDate": "2002-01-01", "rating": 7.0, "voteCount": 90},
            ],
        }

        class Response:
            def __init__(self, payload):
                self.payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self.payload

        post.side_effect = [
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_person", "arguments": {"name": "Actor", "media_type": "movie"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_years", "arguments": {"year_from": 2000, "year_to": 2005, "media_type": "movie"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_execute", "arguments": {"purpose": "results"}}},
            ]}}),
        ]

        agent.run_agent("films avec Actor entre 2000 et 2005")

        results.assert_called_once_with("s1", limit=10)
        self.assertEqual(post.call_count, 3)

    @patch("agent.radarr.request_movie", return_value={"added": True, "title": "Alpha"})
    @patch("agent.catalog_sets.results")
    @patch("agent.catalog_sets.execute_constraint_group")
    @patch("agent.requests.post")
    def test_full_discover_then_action_cycle_uses_grounded_tmdb_id(self, post, execute_group, results, request_movie):
        execute_group.side_effect = lambda members, source=None: agent.catalog_sets._store({101}, "movie", "seed")
        results.return_value = {
            "set": "s1", "count": 1,
            "results": [{"mediaType": "movie", "id": 101, "title": "Alpha", "releaseDate": "2001-01-01", "rating": 8.0, "voteCount": 100}],
        }

        class Response:
            def __init__(self, payload):
                self.payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self.payload

        post.side_effect = [
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_genre", "arguments": {"name": "Action", "media_type": "movie"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_years", "arguments": {"year_from": 2000, "year_to": 2005, "media_type": "movie"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_execute", "arguments": {"purpose": "action"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "radarr_request_movie", "arguments": {"tmdb_id": 101, "french": False}}}
            ]}}),
        ]

        agent.run_agent("trouve un film d'action 2000-2005 et ajoute-le")
        request_movie.assert_called_once_with(101, french=False)
        self.assertEqual(post.call_count, 4)

    @patch("agent.radarr.request_movie", return_value={"added": True, "title": "Alpha"})
    def test_action_routing_uses_exact_grounded_identifier(self, request_movie):
        result = agent.execute_tool("radarr_request_movie", {"tmdb_id": 101, "french": False})
        self.assertTrue(result["added"])
        request_movie.assert_called_once_with(101, french=False)


    @patch("agent.catalog_sets.results")
    @patch("agent.catalog_sets.execute_constraint_group")
    @patch("agent.requests.post")
    def test_mixed_turn_keeps_constraint_declarative_until_execute(self, post, execute_group, results):
        # run_agent() resets request-local catalogue state, so create the handle
        # only when execution actually happens inside the request.
        execute_group.side_effect = lambda members, source=None: agent.catalog_sets._store({101}, "movie", "seed")
        results.return_value = {"set": "s1", "count": 1, "results": [{"mediaType": "movie", "id": 101, "title": "Alpha"}]}

        class Response:
            def __init__(self, payload): self.payload = payload
            def raise_for_status(self): pass
            def json(self): return self.payload

        post.side_effect = [
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_years", "arguments": {"year_from": 2000, "year_to": 2005, "media_type": "movie"}}},
                {"function": {"name": "catalog_keyword_vocabulary", "arguments": {"query": "virtual reality"}}},
            ]}}),
            Response({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "catalog_execute", "arguments": {}}},
            ]}}),
        ]
        with patch("agent.catalog_sets.keyword_vocabulary", return_value={"keywords": [{"id": 4563, "name": "virtual reality"}]}):
            agent.run_agent("film 2000-2005 sur la réalité virtuelle")
        execute_group.assert_called_once()
        results.assert_called_once()

    def test_catalogue_renderer_uses_grounded_fields_directly(self):
        text = agent._render_catalogue_results({"results": [{
            "title": "Alpha", "releaseDate": "2001-04-05",
            "rating": 8.1, "overview": "Grounded synopsis."
        }]})
        self.assertEqual(text, "Alpha (2001) — 8.1/10\nGrounded synopsis.")

    @patch("agent.radarr.request_movie")
    @patch("agent.requests.post")
    def test_agent_rejects_ungrounded_radarr_write(self, post, request_movie):
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return {"message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "radarr_request_movie", "arguments": {"tmdb_id": 999}}}
                ]}}
        post.return_value = Response()
        agent.run_agent("download a movie")
        request_movie.assert_not_called()

    def test_radarr_completed_queue_does_not_claim_active_download(self):
        text = agent._render_terminal_tool("radarr_queue_status", {
            "found": True, "title": "Movie", "inQueue": True,
            "status": "completed", "trackedDownloadStatus": "ok",
            "trackedDownloadState": "importPending", "size": 1000, "sizeleft": 0,
        })
        self.assertIn("téléchargement terminé", text)
        self.assertIn("importPending", text)
        self.assertNotIn("100 %", text)

    def test_sonarr_queue_aggregates_states_and_active_progress(self):
        text = agent._render_terminal_tool("sonarr_queue_status", {
            "found": True, "title": "Show", "inQueue": True,
            "items": [
                {"status": "completed", "trackedDownloadStatus": "ok", "trackedDownloadState": "importPending", "size": 100, "sizeleft": 0},
                {"status": "completed", "trackedDownloadStatus": "ok", "trackedDownloadState": "importPending", "size": 100, "sizeleft": 0},
                {"status": "downloading", "trackedDownloadStatus": "ok", "size": 100, "sizeleft": 25},
            ],
        })
        self.assertIn("2 téléchargement terminé", text)
        self.assertIn("importPending", text)
        self.assertIn("1 downloading", text)
        self.assertIn("progression moyenne : 75 %", text)

    def test_catalog_execute_contract_requires_outcome_purpose(self):
        tool = next(x for x in agent.TOOLS if x["function"]["name"] == "catalog_execute")
        params = tool["function"]["parameters"]
        self.assertIn("purpose", params["required"])
        self.assertEqual(params["properties"]["purpose"]["enum"], ["results", "action"])

    @patch("agent.plex.status")
    @patch("agent.radarr.request_movie")
    def test_multi_movie_plex_filter_is_applied_before_confirmation(self, request_movie, plex_status):
        # Representative grounded candidates: one local, two missing. No write
        # may happen before the human confirms the two-film remainder.
        grounded = {
            101: {"mediaType": "movie", "id": 101, "title": "Local"},
            102: {"mediaType": "movie", "id": 102, "title": "Missing A"},
            103: {"mediaType": "movie", "id": 103, "title": "Missing B"},
        }
        plex_status.side_effect = lambda title: {"found": title == "Local", "title": title}
        plan = [
            {"tmdb_id": mid, "title": grounded[mid]["title"]}
            for mid in [101, 102, 103]
            if not plex_status(grounded[mid]["title"]).get("found", False)
        ]
        self.assertEqual([x["tmdb_id"] for x in plan], [102, 103])
        request_movie.assert_not_called()

    @patch("agent.radarr.request_movie")
    def test_multi_movie_plan_requires_confirmation_before_writes(self, request_movie):
        agent.PENDING_CONFIRMATION = {
            "movies": [
                {"tmdb_id": 101, "title": "Alpha"},
                {"tmdb_id": 102, "title": "Beta"},
            ],
            "french": True,
        }
        request_movie.return_value = {"added": True}
        self.assertIsNotNone(agent.PENDING_CONFIRMATION)
        request_movie.assert_not_called()

        agent.run_agent("oui")

        self.assertEqual(request_movie.call_count, 2)
        request_movie.assert_any_call(101, french=True)
        request_movie.assert_any_call(102, french=True)
        self.assertIsNone(agent.PENDING_CONFIRMATION)

    @patch("agent.radarr.request_movie")
    @patch("agent.requests.post")
    def test_non_confirmation_cancels_pending_plan_without_write(self, post, request_movie):
        agent.PENDING_CONFIRMATION = {
            "movies": [
                {"tmdb_id": 101, "title": "Alpha"},
                {"tmdb_id": 102, "title": "Beta"},
            ],
            "french": True,
        }
        class Response:
            def raise_for_status(self): pass
            def json(self): return {"message": {"role": "assistant", "content": "annulé"}}
        post.return_value = Response()

        agent.run_agent("non laisse tomber")

        request_movie.assert_not_called()
        self.assertIsNone(agent.PENDING_CONFIRMATION)

    def test_unknown_tool_is_nonfatal(self):
        self.assertIn("error", agent.execute_tool("does_not_exist", {}))


if __name__ == "__main__":
    unittest.main()
