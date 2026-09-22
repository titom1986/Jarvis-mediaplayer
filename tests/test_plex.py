import unittest
from unittest.mock import patch

from tools import plex


class PlexTests(unittest.TestCase):
    def test_status_exposes_watched_state(self):
        payload = {
            "MediaContainer": {
                "Hub": [{
                    "Metadata": [
                        {"title": "Seen", "type": "movie", "year": 1995, "ratingKey": "1", "viewCount": 2},
                        {"title": "Unseen", "type": "movie", "year": 1996, "ratingKey": "2"},
                    ]
                }]
            }
        }

        response = unittest.mock.Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with patch.object(plex, "PLEX_TOKEN", "test-token"), patch("tools.plex.requests.get", return_value=response):
            seen = plex.status("Seen")
            unseen = plex.status("Unseen")

        self.assertTrue(seen["results"][0]["watched"])
        self.assertEqual(seen["results"][0]["viewCount"], 2)
        self.assertFalse(unseen["results"][0]["watched"])
        self.assertEqual(unseen["results"][0]["viewCount"], 0)


if __name__ == "__main__":
    unittest.main()
