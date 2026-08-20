"""OpenRouter routing-shim tests; no network or credentials required."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.config import load_config
from src.constants import CONFIG_PATH
from src.models import Problem
from src.openrouter_routing import (
    DEEPSEEK_V4_FLASH_0731,
    GLM47_FP8_ALIAS,
    route_for,
)
from src.run import run_checkpoint_identity
from src.solver import provider_env, provider_model_name, provider_transport_policy


class OpenRouterRoutingTests(unittest.TestCase):
    def test_glm47_alias_is_exact_and_reproducible(self) -> None:
        expected = {
            "model": "z-ai/glm-4.7",
            "provider": {
                "only": ["streamlake/fp8"],
                "allow_fallbacks": False,
                "require_parameters": False,
                "quantizations": ["fp8"],
                "max_price": {"prompt": 0.48, "completion": 1.76},
            },
        }
        self.assertEqual(route_for(GLM47_FP8_ALIAS), expected)
        self.assertEqual(provider_model_name(GLM47_FP8_ALIAS), GLM47_FP8_ALIAS)
        self.assertEqual(
            provider_transport_policy(GLM47_FP8_ALIAS),
            {"policy": "openrouter_frozen_route_v1", "route": expected},
        )

    def test_deepseek_v4_flash_uses_only_frozen_fast_cheap_providers(self) -> None:
        expected = {
            "model": DEEPSEEK_V4_FLASH_0731,
            "provider": {
                "only": ["relace", "baidu", "streamlake", "deepinfra"],
                "allow_fallbacks": True,
                "require_parameters": False,
                "sort": "throughput",
                "max_price": {"prompt": 0.08, "completion": 0.18},
            },
        }
        self.assertEqual(route_for(DEEPSEEK_V4_FLASH_0731), expected)
        self.assertEqual(
            provider_transport_policy(DEEPSEEK_V4_FLASH_0731),
            {"policy": "openrouter_frozen_route_v1", "route": expected},
        )

    def test_unknown_openrouter_model_has_no_frozen_route(self) -> None:
        self.assertIsNone(route_for("vendor/unconfigured-model"))

    def test_shim_injects_route_and_rejects_model_substitution(self) -> None:
        from src.openrouter_proxy import is_messages_path, routed_body, upstream_path

        original = json.dumps(
            {"model": GLM47_FP8_ALIAS, "messages": [{"role": "user", "content": "x"}]}
        ).encode()
        routed = json.loads(routed_body(original, GLM47_FP8_ALIAS))
        expected_route = route_for(GLM47_FP8_ALIAS)
        assert expected_route is not None
        self.assertEqual(routed["model"], "z-ai/glm-4.7")
        self.assertEqual(routed["provider"], expected_route["provider"])
        with self.assertRaisesRegex(ValueError, "model substitution rejected"):
            routed_body(b'{"model":"some/other-model"}', GLM47_FP8_ALIAS)
        self.assertEqual(
            upstream_path("/api/v1/messages?beta=true&trace=yes"),
            "/api/v1/messages?trace=yes",
        )
        self.assertTrue(is_messages_path("/api/v1/messages?beta=true"))
        self.assertFalse(is_messages_path("/api/v1/chat/completions"))

    def test_shim_injects_deepseek_provider_pool(self) -> None:
        from src.openrouter_proxy import routed_body

        original = json.dumps(
            {
                "model": DEEPSEEK_V4_FLASH_0731,
                "messages": [{"role": "user", "content": "x"}],
            }
        ).encode()
        routed = json.loads(routed_body(original, DEEPSEEK_V4_FLASH_0731))
        expected_route = route_for(DEEPSEEK_V4_FLASH_0731)
        assert expected_route is not None
        self.assertEqual(routed["model"], DEEPSEEK_V4_FLASH_0731)
        self.assertEqual(routed["provider"], expected_route["provider"])

    def test_frozen_route_is_part_of_checkpoint_identity(self) -> None:
        for model in (GLM47_FP8_ALIAS, DEEPSEEK_V4_FLASH_0731):
            with self.subTest(model=model):
                config = replace(load_config(CONFIG_PATH), model=model)
                arm = config.arms["baseline"]
                problem = Problem(
                    "test", "statement", "combinatorics", None, None, None
                )
                identity = run_checkpoint_identity(config, arm, problem, 1)
                self.assertEqual(
                    identity["provider_transport_policy"],
                    provider_transport_policy(model),
                )

    @patch.dict(
        "os.environ",
        {"HARNESS_OPENROUTER_PROXY_URL": "http://127.0.0.1:8787/api"},
    )
    def test_openrouter_session_uses_loopback_shim(self) -> None:
        env = provider_env(GLM47_FP8_ALIAS, "secret")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8787/api")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "secret")

    def test_proxy_suppresses_success_access_logs(self) -> None:
        from src.openrouter_proxy import RoutingProxyHandler

        handler = object.__new__(RoutingProxyHandler)
        with self.assertNoLogs("openrouter_proxy", level="INFO"):
            handler.log_message(
                '"%s" %s %s', "POST /api/v1/messages", "200", "-"
            )

    def test_proxy_suppresses_normal_client_connection_reset(self) -> None:
        from http.server import BaseHTTPRequestHandler

        from src.openrouter_proxy import RoutingProxyHandler

        handler = object.__new__(RoutingProxyHandler)
        with patch.object(
            BaseHTTPRequestHandler,
            "handle",
            side_effect=ConnectionResetError("client closed keep-alive socket"),
        ):
            handler.handle()
        self.assertTrue(handler.close_connection)

    def test_proxy_preserves_failed_access_logs(self) -> None:
        from src.openrouter_proxy import RoutingProxyHandler

        handler = object.__new__(RoutingProxyHandler)
        with patch.object(
            RoutingProxyHandler, "address_string", return_value="127.0.0.1"
        ):
            with self.assertLogs("openrouter_proxy", level="WARNING") as captured:
                handler.log_message('"%s" %s %s', "POST /bad", "405", "-")
        self.assertIn('"POST /bad" 405 -', captured.output[0])


if __name__ == "__main__":
    unittest.main()
