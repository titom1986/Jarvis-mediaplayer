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


    def test_realistic_people_first_dataset_filters_before_returning(self):
        # Jeu représentatif : une filmographie assez large, mais seulement quelques
        # films satisfont genre + décennie + exclusion.
        details = {}
        credits = []
        for i in range(1, 166):
            year = 1980 + (i % 45)
            genres = ["Action"]
            keywords = []
            if i in (10, 20, 30, 40):
                year = 1994 + (i // 10)
                genres = ["Science Fiction"]
            if i == 20:
                keywords = ["dystopia"]
            details[i] = {
                "mediaType": "movie", "id": i, "title": f"Movie {i}",
                "releaseDate": f"{year:04d}-01-01", "genres": genres,
                "keywords": keywords, "rating": 7.0 + (i % 10) / 10,
                "voteCount": 100 + i,
            }
            credits.append({"mediaType": "movie", "id": i})

        # Armageddon-like winner: valid SF 90s, not dystopian.
        details[10]["releaseDate"] = "1998-07-01"
        details[10]["rating"] = 6.833
        details[10]["voteCount"] = 8844
        # Other crafted candidates: one dystopian, one outside decade, one valid.
        details[20]["releaseDate"] = "1995-01-01"
        details[30]["releaseDate"] = "2001-01-01"
        details[40]["releaseDate"] = "1997-01-01"

        with patch("tools.media_search.seerr.person_credits", return_value={"results": credits}), \
             patch("tools.media_search.seerr.media_details", side_effect=lambda i, t: details[i]), \
             patch("tools.media_search.seerr.discover") as discover:
            r = media_search.media_search(
                "movie",
                include=[{
                    "people": ["Bruce Willis"],
                    "genres": ["Science Fiction"],
                    "dates": [{"from": 1990, "to": 1999}],
                }],
                exclude=[{"keywords": ["dystopia"], "dates": [{"from": 1990, "to": 1999}]}],
            )

        self.assertFalse(discover.called, "Une recherche avec personne ne doit pas scanner Discover")
        self.assertEqual(r["candidates"], 165)
        self.assertEqual({x["id"] for x in r["results"]}, {10, 40})
        self.assertNotIn(20, {x["id"] for x in r["results"]})
        self.assertNotIn(30, {x["id"] for x in r["results"]})

    def test_realistic_broad_discover_dataset_paginates_without_people(self):
        # Sans personne, Discover est bien la source de candidats et peut être volumineux.
        pages = {
            1: {"results": [{"id": 1}, {"id": 2}], "totalPages": 3},
            2: {"results": [{"id": 3}], "totalPages": 3},
            3: {"results": [], "totalPages": 3},
        }
        with patch("tools.media_search.seerr.discover", side_effect=lambda *a, **k: pages[k["page"]]) as discover:
            r = media_search.media_search(
                "movie",
                include=[{"genres": ["Science Fiction"], "dates": [{"from": 1990, "to": 1999}]}],
            )
        self.assertEqual(discover.call_count, 3)
        self.assertEqual(r["candidates"], 3)


if __name__ == "__main__":
    unittest.main()
