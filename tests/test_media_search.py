import unittest
from unittest.mock import patch

from tools import media_search


DETAILS = {
    1: {"mediaType": "movie", "id": 1, "title": "Time One", "releaseDate": "1995-01-01", "genres": ["Science Fiction"], "keywords": ["time travel"], "rating": 8.1, "voteCount": 1000},
    2: {"mediaType": "movie", "id": 2, "title": "Dystopia", "releaseDate": "1998-01-01", "genres": ["Science Fiction"], "keywords": ["dystopia"], "rating": 9.0, "voteCount": 2000},
    3: {"mediaType": "movie", "id": 3, "title": "Future", "releaseDate": "2022-01-01", "genres": ["Science Fiction"], "keywords": ["time travel"], "rating": 7.5, "voteCount": 500},
}


class MediaSearchTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch("tools.media_search.seerr.genres", return_value=[{"id": 878, "name": "Science Fiction"}]),
            patch("tools.media_search.seerr.search_keyword", side_effect=lambda q: {"results": [{"id": 1 if q == "time travel" else 2, "name": q}]}),
            patch("tools.media_search.seerr.media_details", side_effect=lambda i, t: DETAILS[i]),
            patch("tools.media_search.seerr.search", return_value={"results": [{"mediaType": "person", "id": 99, "title": "Bruce Willis"}]}),
            patch("tools.media_search.seerr.person_credits", return_value={"results": [{"mediaType": "movie", "id": 1}, {"mediaType": "movie", "id": 2}]}),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def test_without_people_and_pagination(self):
        pages = {
            1: {"results": [{"id": 1}], "totalPages": 2},
            2: {"results": [{"id": 2}], "totalPages": 2},
        }
        with patch("tools.media_search.seerr.discover", side_effect=lambda *a, **k: pages[k["page"]]):
            r = media_search.media_search("movie", include=[{"genres": ["Science Fiction"], "dates": [{"from": 1990, "to": 1999}]}])
        self.assertEqual({x["id"] for x in r["results"]}, {1, 2})

    def test_and_inside_group(self):
        with patch("tools.media_search.seerr.discover", return_value={"results": [{"id": 1}, {"id": 2}], "totalPages": 1}):
            r = media_search.media_search("movie", include=[{"genres": ["Science Fiction"], "keywords": ["time travel"], "dates": [{"from": 1990, "to": 1999}]}])
        self.assertEqual([x["id"] for x in r["results"]], [1])

    def test_or_between_groups(self):
        def discover(*a, **k):
            return {"results": [{"id": 1}] if k.get("date_to") == "1999-12-31" else [{"id": 3}], "totalPages": 1}
        with patch("tools.media_search.seerr.discover", side_effect=discover):
            r = media_search.media_search("movie", include=[
                {"genres": ["Science Fiction"], "dates": [{"from": 1990, "to": 1999}]},
                {"genres": ["Science Fiction"], "dates": [{"from": 2020, "to": 2025}]},
            ])
        self.assertEqual({x["id"] for x in r["results"]}, {1, 3})

    def test_exclusion_group(self):
        def discover(*a, **k):
            results = [{"id": 1}, {"id": 2}]
            if k.get("exclude_keyword_ids") == [2]:
                results = [{"id": 1}]
            return {"results": results, "totalPages": 1}
        with patch("tools.media_search.seerr.discover", side_effect=discover) as mocked:
            r = media_search.media_search("movie", include=[{"genres": ["Science Fiction"]}], exclude=[{"keywords": ["dystopia"]}])
        self.assertEqual([x["id"] for x in r["results"]], [1])
        self.assertEqual(mocked.call_args.kwargs["exclude_keyword_ids"], [2])

    def test_people_is_optional_but_supported(self):
        r = media_search.media_search("movie", include=[{"people": ["Bruce Willis"], "keywords": ["time travel"], "dates": [{"from": 1990, "to": 1999}]}])
        self.assertEqual([x["id"] for x in r["results"]], [1])

    def test_rating_sort(self):
        with patch("tools.media_search.seerr.discover", return_value={"results": [{"id": 1}, {"id": 2}], "totalPages": 1}):
            r = media_search.media_search("movie", include=[{"genres": ["Science Fiction"]}])
        self.assertEqual([x["id"] for x in r["results"]], [2, 1])


if __name__ == "__main__":
    unittest.main()
