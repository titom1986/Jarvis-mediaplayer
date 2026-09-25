import unittest
from unittest.mock import patch

from tools import catalog_sets


class CatalogSetTests(unittest.TestCase):
    def setUp(self):
        catalog_sets.reset()

    def test_catalogue_label_normalization_is_lexical_only(self):
        self.assertEqual(
            catalog_sets.media_search._norm("science-fiction"),
            catalog_sets.media_search._norm("Science Fiction"),
        )

    @patch("tools.catalog_sets.media_search._resolve_person")
    @patch("tools.catalog_sets.media_search._credits_ids")
    def test_person_returns_handle_not_ids(self, credits, resolve):
        resolve.return_value = {"id": 42}
        credits.return_value = set(range(165))
        result = catalog_sets.person("Bruce Willis", "movie")
        self.assertEqual(result["set"], "s1")
        self.assertEqual(result["count"], 165)
        self.assertNotIn("ids", result)

    def test_set_algebra_keeps_cardinality_out_of_model_context(self):
        a = catalog_sets._store(range(165), "movie", "person")
        b = catalog_sets._store(range(100, 210), "movie", "genre")
        c = catalog_sets._store(range(120, 150), "movie", "years")
        d = catalog_sets._store(range(130, 140), "movie", "keyword")
        e = catalog_sets._store({132, 135}, "movie", "1998")
        f = catalog_sets._store({133}, "movie", "excluded person")

        abcd = catalog_sets.combine("intersection", [a["set"], b["set"], c["set"], d["set"]])
        excluded = catalog_sets.combine("union", [e["set"], f["set"]])
        final = catalog_sets.subtract(abcd["set"], excluded["set"])

        self.assertEqual(final["count"], 7)
        self.assertNotIn("ids", final)

    @patch("tools.catalog_sets.seerr.media_details")
    @patch("tools.catalog_sets.media_search._discover_ids")
    def test_people_first_refinement_never_broad_scans(self, discover, details):
        base = catalog_sets._store(range(1, 166), "movie", "person")
        def item(media_id, media_type):
            return {
                "id": media_id, "mediaType": media_type,
                "title": f"Movie {media_id}",
                "releaseDate": "1995-01-01" if media_id <= 20 else "2005-01-01",
                "genres": ["Science Fiction"] if media_id <= 40 else ["Action"],
                "keywords": ["time travel"] if media_id <= 10 else [],
            }
        details.side_effect = item

        sf = catalog_sets.genre("Science Fiction", "movie", source=base["set"])
        decade = catalog_sets.years(1990, 1999, "movie", source=sf["set"])
        time_travel = catalog_sets.keyword("time travel", "movie", source=decade["set"])

        self.assertEqual(sf["count"], 40)
        self.assertEqual(decade["count"], 20)
        self.assertEqual(time_travel["count"], 10)
        discover.assert_not_called()
        self.assertEqual(details.call_count, 225)

    @patch("tools.catalog_sets.media_search._resolve_genre_ids", return_value=[878])
    @patch("tools.catalog_sets.media_search._discover_ids")
    def test_first_broad_constraint_can_use_discover(self, discover, resolve_genre):
        discover.return_value = {1, 2, 3}
        result = catalog_sets.genre("Science Fiction", "movie")
        self.assertEqual(result["count"], 3)
        resolve_genre.assert_called_once_with(["Science Fiction"], "movie")
        discover.assert_called_once()

    @patch("tools.catalog_sets.media_search._resolve_genre_ids", return_value=None)
    def test_unresolved_genre_is_error_not_false_empty_set(self, resolve_genre):
        result = catalog_sets.genre("not-a-catalogue-genre", "movie")
        self.assertIn("error", result)
        self.assertNotIn("set", result)

    @patch("tools.catalog_sets.seerr.media_details")
    @patch("tools.catalog_sets.media_search._credits_ids")
    @patch("tools.catalog_sets.media_search._resolve_person")
    def test_representative_end_to_end_set_path(self, resolve, credits, details):
        people = {
            "Bruce Willis": (1, set(range(1, 166))),
            "Scarlett Johansson": (2, {7, 50, 90}),
        }
        resolve.side_effect = lambda name: {"id": people[name][0], "title": name}
        credits.side_effect = lambda person_id, media_type: (
            people["Bruce Willis"][1] if person_id == 1 else people["Scarlett Johansson"][1]
        )

        def item(media_id, media_type):
            return {
                "id": media_id,
                "mediaType": media_type,
                "title": f"Movie {media_id}",
                "releaseDate": (
                    "1998-01-01" if media_id in {5, 6}
                    else "1995-01-01" if media_id <= 20
                    else "2005-01-01"
                ),
                "genres": ["Science Fiction"] if media_id <= 40 else ["Action"],
                "keywords": ["time travel"] if media_id <= 10 else [],
                "rating": 9.0 - media_id / 100,
                "voteCount": 1000 - media_id,
                "overview": "",
            }
        details.side_effect = item

        bruce = catalog_sets.person("Bruce Willis", "movie")
        sf = catalog_sets.genre("Science Fiction", "movie", source=bruce["set"])
        decade = catalog_sets.years(1990, 1999, "movie", source=sf["set"])
        travel = catalog_sets.keyword("time travel", "movie", source=decade["set"])
        year_98 = catalog_sets.years(1998, 1998, "movie", source=travel["set"])
        scarlett = catalog_sets.person("Scarlett Johansson", "movie")
        excluded = catalog_sets.combine("union", [year_98["set"], scarlett["set"]])
        final = catalog_sets.subtract(travel["set"], excluded["set"])
        ranked = catalog_sets.results(final["set"], limit=20)

        self.assertEqual(bruce["count"], 165)
        self.assertEqual(sf["count"], 40)
        self.assertEqual(decade["count"], 20)
        self.assertEqual(travel["count"], 10)
        self.assertEqual(year_98["count"], 2)
        self.assertEqual(scarlett["count"], 3)
        self.assertEqual(final["count"], 7)
        self.assertEqual([x["id"] for x in ranked["results"]], [1, 2, 3, 4, 8, 9, 10])
        self.assertNotIn(5, [x["id"] for x in ranked["results"]])
        self.assertNotIn(6, [x["id"] for x in ranked["results"]])
        self.assertNotIn(7, [x["id"] for x in ranked["results"]])
        # Final tool payload stays compact: semantic filtering fields remain
        # internal, while the catalogue synopsis grounds natural-language descriptions.
        self.assertEqual(
            set(ranked["results"][0]),
            {"mediaType", "id", "title", "releaseDate", "rating", "voteCount", "overview"},
        )
        self.assertEqual(ranked["results"][0]["overview"], "")
        self.assertNotIn("genres", ranked["results"][0])
        self.assertNotIn("keywords", ranked["results"][0])

    @patch("tools.catalog_sets.seerr.search_keyword")
    def test_keyword_vocabulary_exposes_catalogue_labels_without_semantic_choice(self, search_keyword):
        search_keyword.return_value = {"results": [
            {"id": 1, "name": "alien"},
            {"id": 2, "name": "alien invasion"},
            {"id": 3, "name": "extraterrestrial"},
        ]}
        result = catalog_sets.keyword_vocabulary("alien", limit=2)
        self.assertEqual(result, {
            "query": "alien", "count": 2,
            "keywords": [{"id": 1, "name": "alien"}, {"id": 2, "name": "alien invasion"}],
        })
        search_keyword.assert_called_once_with("alien")

    @patch("tools.catalog_sets.seerr.discover")
    @patch("tools.catalog_sets.media_search._resolve_keyword_ids")
    @patch("tools.catalog_sets.seerr.search_keyword")
    def test_grounded_keyword_ids_are_reused_without_label_relookup(self, search_keyword, resolve_keyword, discover):
        search_keyword.return_value = {"results": [
            {"id": 9951, "name": "alien"},
            {"id": 192962, "name": "alien attack"},
        ]}
        catalog_sets.keyword_vocabulary("alien")
        discover.side_effect = [
            {"totalResults": 100, "results": []},
            {"totalResults": 20, "results": []},
        ]

        result = catalog_sets.estimate_constraint("catalog_keyword", {
            "name": "alien", "aliases": ["alien attack"], "media_type": "movie"
        })

        self.assertEqual(result["keyword_ids"], [9951, 192962])
        self.assertEqual(result["count"], 120)
        resolve_keyword.assert_not_called()
        self.assertEqual(discover.call_count, 2)

    @patch("tools.catalog_sets.seerr.media_details")
    def test_keyword_aliases_are_or_alternatives_when_refining(self, details):
        base = catalog_sets._store({1, 2, 3}, "movie", "base")
        payloads = {
            1: {"id": 1, "mediaType": "movie", "keywords": ["alien"]},
            2: {"id": 2, "mediaType": "movie", "keywords": ["extraterrestrial"]},
            3: {"id": 3, "mediaType": "movie", "keywords": ["space travel"]},
        }
        details.side_effect = lambda media_id, media_type: payloads[media_id]
        result = catalog_sets.keyword(
            "extraterrestrials", "movie", source=base["set"],
            aliases=["alien", "extraterrestrial"],
        )
        self.assertEqual(catalog_sets._get(result["set"])["ids"], {1, 2})

    def test_unknown_handle_is_safe_error(self):
        self.assertIn("error", catalog_sets.combine("intersection", ["missing"]))

    def test_mixed_media_types_are_rejected(self):
        a = catalog_sets._store({1}, "movie", "a")
        b = catalog_sets._store({1}, "tv", "b")
        self.assertIn("error", catalog_sets.combine("intersection", [a["set"], b["set"]]))


if __name__ == "__main__":
    unittest.main()
