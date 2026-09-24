import unittest
from unittest.mock import patch

from tools import catalog_sets


class CatalogSetTests(unittest.TestCase):
    def setUp(self):
        catalog_sets.reset()

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

    @patch("tools.catalog_sets.media_search._discover_ids")
    def test_first_broad_constraint_can_use_discover(self, discover):
        discover.return_value = {1, 2, 3}
        result = catalog_sets.genre("Science Fiction", "movie")
        self.assertEqual(result["count"], 3)
        discover.assert_called_once()

    def test_unknown_handle_is_safe_error(self):
        self.assertIn("error", catalog_sets.combine("intersection", ["missing"]))

    def test_mixed_media_types_are_rejected(self):
        a = catalog_sets._store({1}, "movie", "a")
        b = catalog_sets._store({1}, "tv", "b")
        self.assertIn("error", catalog_sets.combine("intersection", [a["set"], b["set"]]))


if __name__ == "__main__":
    unittest.main()
