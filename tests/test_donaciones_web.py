import json
import threading
import unittest
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import donaciones_web as web


SAMPLE = [
    {"id": 1, "valor": "100", "nombre": "Ana", "fecha": "Hace 2 horas", "mensaje": "hola https://x.com/a", "privado": 0},
    {"id": 2, "valor": "8000", "nombre": "Bruno", "fecha": "Hace 2 meses", "mensaje": "chau", "privado": 0},
    {"id": 3, "valor": "500", "nombre": "AnaPlus", "fecha": "Hace 1 día", "mensaje": "", "privado": 1},
]


class BindTests(unittest.TestCase):
    def test_skips_a_busy_port(self):
        blocker = web.ReuseServer(("127.0.0.1", 0), web.Handler)
        busy = blocker.server_address[1]
        try:
            server = web.bind_server("127.0.0.1", busy, web.Handler)
            try:
                self.assertNotEqual(server.server_address[1], busy)
            finally:
                server.server_close()
        finally:
            blocker.server_close()


class PublicReportTests(unittest.TestCase):
    def test_kpis_and_table_slice(self):
        normalized = [web.core.normalize_donation(item) for item in SAMPLE]
        report = web.public_report(normalized, top=5, offset=1, limit=1)
        self.assertEqual(report["cantidad"], 3)
        self.assertEqual(len(report["kpis"]), 6)
        self.assertEqual(report["tabla"]["total"], 3)
        self.assertEqual(len(report["tabla"]["items"]), 1)
        self.assertTrue(report["insights"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        web.STORE.set_donations([web.core.normalize_donation(item) for item in SAMPLE], "test")
        handler = partial(web.Handler, page_user="losherederosdealberdi")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as response:
            return response.status, response.read()

    def test_home_and_filtered_report(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"LosHerederosDeAlberdi", body)
        status, body = self.get("/api/reporte?donante=ana&fecha=semana")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["cantidad"], 2)
        self.assertEqual(data["filtros"]["donante"], "ana")
        status, body = self.get("/api/reporte?min=0&max=0")
        data = json.loads(body)
        self.assertEqual(data["cantidad"], 3)
        self.assertEqual(data["filtros"], {})

    def test_complete_listado_keeps_every_row(self):
        status, body = self.get("/listado")
        self.assertEqual(status, 200)
        self.assertIn(b"3 donaciones", body)
        self.assertIn(b"Ana</td>", body)
        self.assertIn(b"Bruno</td>", body)
        self.assertIn(b"AnaPlus</td>", body)
        status, body = self.get("/api/listado.csv")
        self.assertEqual(status, 200)
        lines = body.decode("utf-8").strip().splitlines()
        self.assertEqual(len(lines), 4)


if __name__ == "__main__":
    unittest.main()
