import json
import unittest
from unittest.mock import Mock

from planner import compile_media_search, extract_intent, validate_intent


EMPTY = {"people": [], "genres": [], "keywords": [], "dates": []}


class PlannerTests(unittest.TestCase):
    def test_compile_cumulative_search_as_one_and_group(self):
        intent = {
            "media_type": "movie", "search": True,
            "required": {
                "people": ["Bruce Willis"],
                "genres": ["Science Fiction"],
                "keywords": [],
                "dates": [{"from": 1990, "to": 1999}],
            },
            "forbidden": {**EMPTY, "keywords": ["dystopia"]},
            "alternatives": [],
        }
        args = compile_media_search(intent)
        self.assertEqual(len(args["include"]), 1)
        self.assertEqual(args["include"][0]["people"], ["Bruce Willis"])
        self.assertEqual(args["exclude"][0]["keywords"], ["dystopia"])

    def test_compile_explicit_alternatives_as_or_groups(self):
        a = {**EMPTY, "people": ["Bruce Willis"]}
        b = {**EMPTY, "people": ["Brad Pitt"]}
        intent = {
            "media_type": "movie", "search": True,
            "required": EMPTY, "forbidden": EMPTY,
            "alternatives": [a, b],
        }
        args = compile_media_search(intent)
        self.assertEqual(args["include"], [a, b])

    def test_extractor_uses_json_schema_and_temperature_zero(self):
        payload = {
            "media_type": "movie", "search": True,
            "required": EMPTY, "forbidden": EMPTY, "alternatives": [],
            "recommend_by_rating": False, "avoid_watched": False,
            "download": False, "french_download": False,
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"message": {"content": json.dumps(payload)}}
        post = Mock(return_value=response)

        result = extract_intent("un film", post=post)

        self.assertEqual({k: v for k, v in result.items() if k != "_perf"}, payload)
        self.assertIn("_perf", result)
        body = post.call_args.kwargs["json"]
        self.assertIsInstance(body["format"], dict)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["keep_alive"], "30m")

    def test_rejects_duplicate_semantic_classification(self):
        intent = {
            "media_type": "movie", "search": True,
            "required": {
                "people": ["Bruce Willis"],
                "genres": ["Science Fiction"],
                "keywords": ["Bruce Willis"],
                "dates": [{"from": 1990, "to": 1999}],
            },
            "forbidden": EMPTY, "alternatives": [],
            "recommend_by_rating": True, "avoid_watched": True,
            "download": False, "french_download": False,
        }
        with self.assertRaises(ValueError):
            validate_intent(intent)

    def test_accepts_exclusive_semantic_classification(self):
        intent = {
            "media_type": "movie", "search": True,
            "required": {
                "people": ["Bruce Willis"],
                "genres": ["Science Fiction"],
                "keywords": [],
                "dates": [{"from": 1990, "to": 1999}],
            },
            "forbidden": {**EMPTY, "keywords": ["dystopia"]},
            "alternatives": [],
            "recommend_by_rating": True, "avoid_watched": True,
            "download": False, "french_download": False,
        }
        self.assertIs(validate_intent(intent), intent)


if __name__ == "__main__":
    unittest.main()
