"""Model downloads against a local server. No real weights."""
import json
import os
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import app
import model_fetch


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.hits += 1
        route = self.server.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        content_type, payload = route
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


class _Server:
    def __init__(self, routes):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.hits = 0
        self.httpd.routes = routes
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def close(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()


def _wav(path):
    with wave.open(path, "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 4410)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        self._saved = {
            "VOCAL_ONNX_URL": model_fetch.VOCAL_ONNX_URL,
            "VOCAL_ONNX_MIN_BYTES": model_fetch.VOCAL_ONNX_MIN_BYTES,
            "HUBERT_CONFIG_URL": model_fetch.HUBERT_CONFIG_URL,
            "HUBERT_CONFIG_MIN_BYTES": model_fetch.HUBERT_CONFIG_MIN_BYTES,
            "HUBERT_WEIGHTS_URL": model_fetch.HUBERT_WEIGHTS_URL,
            "HUBERT_WEIGHTS_MIN_BYTES": model_fetch.HUBERT_WEIGHTS_MIN_BYTES,
            "RMVPE_URL": model_fetch.RMVPE_URL,
            "RMVPE_MIN_BYTES": model_fetch.RMVPE_MIN_BYTES,
        }
        self.payload = b"M" * 200
        self.config = json.dumps({"model_type": "hubert", "hidden_size": 768}).encode()
        self.server = _Server(
            {
                "/onnx": ("application/octet-stream", self.payload),
                "/html": ("text/html", b"<!DOCTYPE html><html>no</html>"),
                "/tiny": ("application/octet-stream", b"short"),
                "/config": ("application/json", self.config),
                "/weights": ("application/octet-stream", self.payload),
                "/rmvpe": ("application/octet-stream", self.payload),
                "/badjson": ("application/json", b'{"model_type":"bert"}'),
            }
        )
        base = self.server.url
        model_fetch.VOCAL_ONNX_URL = base + "/onnx"
        model_fetch.VOCAL_ONNX_MIN_BYTES = len(self.payload)
        model_fetch.HUBERT_CONFIG_URL = base + "/config"
        model_fetch.HUBERT_CONFIG_MIN_BYTES = 16
        model_fetch.HUBERT_WEIGHTS_URL = base + "/weights"
        model_fetch.HUBERT_WEIGHTS_MIN_BYTES = len(self.payload)
        model_fetch.RMVPE_URL = base + "/rmvpe"
        model_fetch.RMVPE_MIN_BYTES = len(self.payload)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(model_fetch, name, value)
        self.server.close()
        self._tmpdir.cleanup()

    def test_vocal_sidecar_uses_voc_ft_fft(self):
        config = model_fetch.VOCAL_SIDECAR
        self.assertEqual(config["dim_f"], 3072)
        self.assertEqual(config["dim_t"], 256)
        self.assertEqual(config["n_fft"], 6144)
        self.assertGreaterEqual(config["n_fft"] // 2 + 1, config["dim_f"])
        self.assertEqual(config["compensate"], 1.021)

    def test_download_writes_sidecar_and_skips_the_second_time(self):
        mdx = os.path.join(self.root, "mdx")
        notes = []
        path, downloaded = model_fetch.ensure_vocal_model(
            mdx, on_progress=lambda frac, message: notes.append(message)
        )
        self.assertTrue(downloaded)
        self.assertTrue(path.endswith("UVR-MDX-NET-Voc_FT.onnx"))
        self.assertEqual(os.path.getsize(path), len(self.payload))
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        self.assertEqual(sidecar["n_fft"], 6144)
        self.assertTrue(any(note.startswith("Bajando") for note in notes))
        hits = self.server.httpd.hits
        _path, again = model_fetch.ensure_vocal_model(mdx)
        self.assertFalse(again)
        self.assertEqual(self.server.httpd.hits, hits)
        self.assertFalse(os.path.exists(path + ".part"))

    def test_html_and_truncated_responses_are_rejected(self):
        dest = os.path.join(self.root, "bad.bin")
        with self.assertRaises(model_fetch.DownloadError) as caught:
            model_fetch.download_file(
                self.server.url + "/html",
                dest,
                min_bytes=1,
                label="el modelo",
            )
        self.assertIn("no es el archivo", str(caught.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))
        with self.assertRaises(model_fetch.DownloadError) as caught:
            model_fetch.download_file(
                self.server.url + "/tiny",
                dest,
                min_bytes=1000,
                label="el modelo",
            )
        self.assertIn("incompleto", str(caught.exception))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_refused_connection_stays_in_spanish(self):
        dest = os.path.join(self.root, "gone.bin")
        with self.assertRaises(model_fetch.DownloadError) as caught:
            model_fetch.download_file(
                "http://127.0.0.1:1/no",
                dest,
                min_bytes=1,
                label="el modelo",
            )
        self.assertIn("conexión", str(caught.exception))
        self.assertNotIn("127.0.0.1", str(caught.exception))

    def test_rvc_support_downloads_once_and_rejects_a_wrong_config(self):
        rvc = os.path.join(self.root, "rvc")
        self.assertTrue(model_fetch.ensure_rvc_support(rvc))
        hubert = os.path.join(rvc, "hubert_base")
        self.assertTrue(os.path.isfile(os.path.join(hubert, "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(hubert, "model.safetensors")))
        self.assertGreaterEqual(
            os.path.getsize(os.path.join(rvc, "rmvpe.pt")),
            len(self.payload),
        )
        hits = self.server.httpd.hits
        self.assertFalse(model_fetch.ensure_rvc_support(rvc))
        self.assertEqual(self.server.httpd.hits, hits)

        model_fetch.HUBERT_CONFIG_URL = self.server.url + "/badjson"
        os.remove(os.path.join(hubert, "config.json"))
        with self.assertRaises(model_fetch.DownloadError):
            model_fetch.ensure_rvc_support(rvc)
        self.assertFalse(os.path.isfile(os.path.join(hubert, "config.json")))

    def test_existing_pytorch_weights_are_not_downloaded_again(self):
        rvc = os.path.join(self.root, "rvc")
        folder = os.path.join(rvc, "hubert_base")
        os.makedirs(folder)
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as handle:
            handle.write(self.config.decode())
        with open(os.path.join(folder, "pytorch_model.bin"), "wb") as handle:
            handle.write(self.payload)
        with open(os.path.join(rvc, "rmvpe.pt"), "wb") as handle:
            handle.write(self.payload)
        hits = self.server.httpd.hits
        self.assertFalse(model_fetch.ensure_rvc_support(rvc))
        self.assertEqual(self.server.httpd.hits, hits)
        self.assertFalse(os.path.isfile(os.path.join(folder, "model.safetensors")))


class SeparationChoiceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        self._mdx = app.MDX_DIR
        self._clean = app.CLEAN_DIR
        app.MDX_DIR = os.path.join(self.root, "mdx")
        app.CLEAN_DIR = os.path.join(self.root, "clean")
        os.makedirs(app.MDX_DIR)

    def tearDown(self):
        app.MDX_DIR = self._mdx
        app.CLEAN_DIR = self._clean
        self._tmpdir.cleanup()

    def test_missing_onnx_downloads_and_selects_it(self):
        expected = os.path.join(app.MDX_DIR, "UVR-MDX-NET-Voc_FT.onnx")

        def fake(directory, on_progress=None):
            self.assertEqual(directory, app.MDX_DIR)
            with open(expected, "wb") as handle:
                handle.write(b"onnx")
            return expected, True

        with mock.patch.object(model_fetch, "ensure_vocal_model", side_effect=fake):
            choice, update = app.prepare_separation("")
        self.assertEqual(choice, expected)
        self.assertEqual(update["value"], expected)

    def test_local_choice_stays_local_when_an_onnx_exists(self):
        with open(os.path.join(app.MDX_DIR, "otro.onnx"), "wb") as handle:
            handle.write(b"onnx")
        with mock.patch.object(model_fetch, "ensure_vocal_model") as fetch:
            choice, _update = app.prepare_separation("")
        fetch.assert_not_called()
        self.assertEqual(choice, "")

    def test_local_note_does_not_mention_a_folder(self):
        src = os.path.join(self.root, "mix.wav")
        _wav(src)
        _vocal, _inst, note = app.separate_to_files(src, "")
        self.assertIn("separación local", note)
        self.assertNotIn("Library", note)
        self.assertNotIn(app.MDX_DIR, note)

    def test_failed_download_still_separates_locally(self):
        src = os.path.join(self.root, "mix.wav")
        _wav(src)

        def fail(*_args, **_kwargs):
            raise model_fetch.DownloadError("No pude bajar el modelo de separación.")

        with mock.patch.object(model_fetch, "ensure_vocal_model", side_effect=fail):
            _source, vocal, instrumental, _converted, _remix, message, _mdx = app.on_separate(
                src, "", "ambas", "", progress=None
            )
        self.assertTrue(os.path.isfile(vocal))
        self.assertTrue(os.path.isfile(instrumental))
        self.assertIn("separación local", message)
        self.assertNotIn("Library", message)
        self.assertNotIn(self.root, message)

    def test_refresh_and_fetch_status_do_not_print_paths(self):
        with mock.patch.object(app, "rvc_model_choices", return_value=[]):
            note = app.on_refresh()[-1]
        self.assertNotIn("Library", note)
        self.assertNotIn("/", note)
        self.assertIn(".pth", note)
        self.assertIn(
            ".pth",
            model_fetch.fetch_status(True),
        )
        self.assertNotIn("Library", model_fetch.fetch_status(False))


class UiCopyTests(unittest.TestCase):
    def test_blocks_do_not_send_people_to_copy_models(self):
        demo = app.build_server()
        texts = []
        for block in demo.blocks.values():
            for attr in ("value", "label", "info", "placeholder"):
                value = getattr(block, attr, None)
                if isinstance(value, str):
                    texts.append(value)
        joined = "\n".join(texts)
        self.assertIn("Bajá los modelos", joined)
        self.assertIn("Abrí la carpeta de voces", joined)
        self.assertNotIn("Application Support", joined)
        self.assertNotIn("no los descarga", joined)
        self.assertNotIn("Library", joined)

    def test_open_voices_does_not_print_a_path(self):
        with mock.patch.object(app, "_open_directory"):
            message = app.on_open_voices()
        self.assertIn(".pth", message)
        self.assertNotIn("Library", message)
        self.assertNotIn("/", message)
