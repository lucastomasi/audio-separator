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
        self.assertIs(kwargs.get("theme"), app.APP_THEME)
        self.assertEqual(kwargs.get("css"), app.UI_CSS)
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
        os.environ.pop("AUDIO_SEPARATOR_HOST", None)

    def test_lan_stays_closed_until_the_host_is_opened(self):
        os.environ[app.TOKEN_ENV] = "b" * 32
        token = os.environ[app.TOKEN_ENV]
        lan = self._request(host="192.168.1.20:7860", query={app.TOKEN_QUERY: token})
        self.assertIsNone(app.auth_dependency(lan))
        os.environ["AUDIO_SEPARATOR_HOST"] = "0.0.0.0"
        self.assertEqual(app.auth_dependency(lan), "local")
        self.assertIsNone(app.auth_dependency(self._request(host="192.168.1.20:7860")))
        self.assertEqual(app.launch_kwargs()["server_name"], "0.0.0.0")

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

    def test_splash_follows_system_dark(self):
        self.assertIn("prefers-color-scheme: dark", desktop.SPLASH)
        self.assertIn("#1C1C1E", desktop.SPLASH)

    def test_ready_rejects_unauthorized_listener(self):
        from email.message import Message

        error = desktop.urllib.error.HTTPError(
            "http://127.0.0.1:1/", 401, "no", Message(), None
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            self.assertFalse(
                desktop.our_server_ready("http://127.0.0.1:1/?access_token=x", "x", timeout=0.2)
            )

    def test_resolve_server_skips_foreign_listener(self):
        class ImmediateThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target
                self._args = args

            def start(self):
                if self._target:
                    self._target(*self._args)

        os.environ["AUDIO_SEPARATOR_PORT"] = "7860"
        self.addCleanup(lambda: os.environ.pop("AUDIO_SEPARATOR_PORT", None))
        with mock.patch("app_env.pick_port") as pick:
            pick.side_effect = [7860, 7861]
            with mock.patch.object(desktop, "port_open", side_effect=[True, False]):
                with mock.patch.object(desktop, "our_server_ready", return_value=False):
                    with mock.patch.object(desktop, "start_server") as start:
                        with mock.patch.object(desktop.threading, "Thread", ImmediateThread):
                            port, url, already = desktop.resolve_server(
                                "tok", probe_timeout=0.1
                            )
        self.assertEqual(port, 7861)
        self.assertFalse(already)
        self.assertIn("7861", url)
        start.assert_called_once_with(7861)

    def test_resolve_server_reuses_when_token_ok(self):
        os.environ["AUDIO_SEPARATOR_PORT"] = "7900"
        self.addCleanup(lambda: os.environ.pop("AUDIO_SEPARATOR_PORT", None))
        with mock.patch("app_env.pick_port", return_value=7900):
            with mock.patch.object(desktop, "port_open", return_value=True):
                with mock.patch.object(desktop, "our_server_ready", return_value=True):
                    with mock.patch.object(desktop, "start_server") as start:
                        port, url, already = desktop.resolve_server("tok", probe_timeout=0.1)
        self.assertEqual(port, 7900)
        self.assertTrue(already)
        start.assert_not_called()

    def test_share_flag_removed(self):
        import inspect

        source = inspect.getsource(app)
        self.assertNotIn('--share', source)
        self.assertNotIn('add_argument("--share"', source)
        self.assertFalse(app.launch_kwargs()["share"])
