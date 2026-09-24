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

    def test_unknown_handle_is_safe_error(self):
        self.assertIn("error", catalog_sets.combine("intersection", ["missing"]))

    def test_mixed_media_types_are_rejected(self):
        a = catalog_sets._store({1}, "movie", "a")
        b = catalog_sets._store({1}, "tv", "b")
        self.assertIn("error", catalog_sets.combine("intersection", [a["set"], b["set"]]))


if __name__ == "__main__":
    unittest.main()
