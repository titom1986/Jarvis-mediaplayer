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


def tool_response(name, arguments):
    return Response({"tool_calls": [{"function": {"name": name, "arguments": arguments}}]})


class SemanticRouterTests(unittest.TestCase):
    @patch("semantic_router.requests.post")
    def test_named_media_action_is_two_stage_semantic_route(self, post):
        post.side_effect = [
            tool_response("route_media_reference", {"kind": "named_media"}),
            tool_response("route_named_media_intent", {"intent": "action"}),
        ]
        self.assertEqual(semantic_router.route("Tu pourrais me choper Silo ?"), "named_action")
        self.assertEqual(post.call_count, 2)
        first = post.call_args_list[0].kwargs["json"]
        second = post.call_args_list[1].kwargs["json"]
        self.assertEqual(len(first["tools"]), 1)
        self.assertEqual(len(second["tools"]), 1)
        self.assertNotIn("Silo", str(first["tools"]))
        self.assertNotIn("Silo", str(second["tools"]))

    @patch("semantic_router.requests.post")
    def test_named_media_status_is_two_stage_semantic_route(self, post):
        post.side_effect = [
            tool_response("route_media_reference", {"kind": "named_media"}),
            tool_response("route_named_media_intent", {"intent": "status"}),
        ]
        self.assertEqual(semantic_router.route("J'ai Silo sur Plex ?"), "status")

    @patch("semantic_router.requests.post")
    def test_discovery_stops_after_first_stage(self, post):
        post.return_value = tool_response("route_media_reference", {"kind": "discovery"})
        self.assertEqual(semantic_router.route("Un film de SF des années 90"), "discovery")
        self.assertEqual(post.call_count, 1)

    @patch("semantic_router.requests.post")
    def test_transfer_stops_after_first_stage(self, post):
        post.return_value = tool_response("route_media_reference", {"kind": "transfer"})
        self.assertEqual(semantic_router.route("Le téléchargement de Silo en est où ?"), "download_status")
        self.assertEqual(post.call_count, 1)

    @patch("semantic_router.requests.post")
    def test_router_falls_back_when_first_stage_does_not_classify(self, post):
        post.return_value = Response({"content": "uncertain"})
        self.assertIsNone(semantic_router.route("phrase ambiguë"))

    @patch("semantic_router.requests.post")
    def test_router_falls_back_when_named_intent_does_not_classify(self, post):
        post.side_effect = [
            tool_response("route_media_reference", {"kind": "named_media"}),
            Response({"content": "uncertain"}),
        ]
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
