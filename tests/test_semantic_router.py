import unittest
from unittest.mock import patch

import semantic_router


class Response:
    def __init__(self, message):
        self.message = message
    def raise_for_status(self):
        pass
    def json(self):
        return {"message": self.message}


class SemanticRouterTests(unittest.TestCase):
    @patch("semantic_router.requests.post")
    def test_router_uses_semantic_family_tool_call(self, post):
        post.return_value = Response({
            "tool_calls": [{"function": {
                "name": "route_media_request",
                "arguments": {"family": "named_action"},
            }}]
        })
        self.assertEqual(semantic_router.route("Tu pourrais me choper Silo ?"), "named_action")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(len(payload["tools"]), 1)
        self.assertNotIn("Silo", str(payload["tools"]))

    @patch("semantic_router.requests.post")
    def test_router_falls_back_when_model_does_not_classify(self, post):
        post.return_value = Response({"content": "uncertain"})
        self.assertIsNone(semantic_router.route("phrase ambiguë"))

    def test_family_filter_never_depends_on_phrase(self):
        tools = [
            {"function": {"name": "catalog_title"}},
            {"function": {"name": "catalog_person"}},
            {"function": {"name": "radarr_request_movie"}},
            {"function": {"name": "radarr_queue_status"}},
        ]
        selected = semantic_router.tools_for_family(tools, "named_action")
        self.assertEqual(
            {x["function"]["name"] for x in selected},
            {"catalog_title", "radarr_request_movie"},
        )

    def test_unknown_family_returns_full_toolset(self):
        tools = [{"function": {"name": "catalog_title"}}]
        self.assertIs(semantic_router.tools_for_family(tools, None), tools)


if __name__ == "__main__":
    unittest.main()
