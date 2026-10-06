"""Local-app security helpers: served paths, errors, ONNX caps, desktop port."""
import os
import tempfile
import unittest
from unittest import mock

import app
import desktop


class LaunchSecurityTests(unittest.TestCase):
    def test_allowed_paths_are_outputs_only(self):
        allowed = [os.path.realpath(path) for path in app._allowed_paths()]
        self.assertNotIn(os.path.realpath(app.__file__.rsplit("/", 1)[0]), allowed)
        self.assertNotIn(os.path.realpath(tempfile.gettempdir()), allowed)
        self.assertTrue(
            any(
                path.endswith("clean_song_output") or path.endswith("Trabajos")
                for path in allowed
            )
        )

    def test_blocked_paths_include_secrets(self):
        blocked = app._blocked_paths()
        self.assertIn("/etc", blocked)
        self.assertTrue(any(path.endswith(".ssh") for path in blocked))

    def test_launch_kwargs_are_locked_down(self):
        kwargs = app.launch_kwargs()
        self.assertFalse(kwargs["share"])
        self.assertFalse(kwargs["show_error"])
        self.assertEqual(kwargs["server_name"], "127.0.0.1")
        self.assertTrue(kwargs.get("strict_cors", True))
        self.assertIs(kwargs["auth_dependency"], app.auth_dependency)
        self.assertNotIn(os.path.realpath(tempfile.gettempdir()), kwargs["allowed_paths"])
        for path in kwargs["allowed_paths"]:
            self.assertNotEqual(os.path.realpath(path), os.path.realpath(os.path.expanduser("~")))

    def test_ui_error_hides_paths(self):
        with self.assertRaises(Exception) as caught:
            app._ui_error(ValueError("falló en /Users/ada/Library/secret"))
        self.assertIn("Algo salió mal.", str(caught.exception))
        self.assertNotIn("/Users/", str(caught.exception))

    def test_ui_error_keeps_safe_spanish(self):
        with self.assertRaises(Exception) as caught:
            app._ui_error(ValueError("Pega un enlace de YouTube."))
        self.assertIn("YouTube", str(caught.exception))

    def test_onnx_rejects_huge_dim(self):
        class Session:
            def get_inputs(self):
                return [type("In", (), {"shape": [1, 4, 8, 16]})()]

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.onnx")
            sidecar = os.path.join(tmp, "m.json")
            with open(model, "wb") as handle:
                handle.write(b"onnx")
            with open(sidecar, "w", encoding="utf-8") as handle:
                handle.write('{"dim_t": 999999, "dim_f": 8, "n_fft": 16, "hop": 2}')
            with self.assertRaises(ValueError):
                app._mdx_config(model, Session())


class AuthTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(app.TOKEN_ENV, None)

    def _request(self, host="127.0.0.1:7860", query=None, cookie=None, header=None, referer=None):
        request = mock.Mock()
        request.headers = {"host": host}
        if header:
            request.headers[app.TOKEN_HEADER] = header
        if referer:
            request.headers["referer"] = referer
        request.query_params = query or {}
        request.cookies = cookie or {}
        return request

    def test_auth_requires_token_and_loopback(self):
        os.environ[app.TOKEN_ENV] = "a" * 32
        token = os.environ[app.TOKEN_ENV]
        self.assertIsNone(app.auth_dependency(self._request()))
        self.assertEqual(
            app.auth_dependency(self._request(query={app.TOKEN_QUERY: token})),
            "local",
        )
        self.assertIsNone(
            app.auth_dependency(
                self._request(host="evil.example", query={app.TOKEN_QUERY: token})
            )
        )


class DesktopTests(unittest.TestCase):
    def test_pick_free_port_is_ephemeral(self):
        port = desktop.pick_free_port()
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)

    def test_ready_rejects_unauthorized_listener(self):
        from email.message import Message

        error = desktop.urllib.error.HTTPError(
            "http://127.0.0.1:1/", 401, "no", Message(), None
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            self.assertFalse(
                desktop.our_server_ready("http://127.0.0.1:1/?access_token=x", "x", timeout=0.2)
            )
