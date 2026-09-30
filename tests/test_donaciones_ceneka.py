import unittest

import donaciones_ceneka as donaciones


class FormatAmountTests(unittest.TestCase):
    def test_whole_pesos_use_space_thousands(self):
        self.assertEqual(donaciones.format_amount("8000.00"), "$ 8 000")
        self.assertEqual(donaciones.format_amount("500.00"), "$ 500")

    def test_decimals_and_missing_values(self):
        self.assertEqual(donaciones.format_amount("12.5"), "$ 12.50")
        self.assertEqual(donaciones.format_amount(None), "$ —")


class FetchLatestTests(unittest.TestCase):
    def test_pages_with_a_fixed_size_and_stops_at_count(self):
        calls = []

        def fetcher(user, page, limit):
            calls.append((user, page, limit))
            start = page * limit
            return [{"id": i, "valor": "100.00", "nombre": f"n{i}", "mensaje": "", "fecha": "hoy", "privado": 0} for i in range(start, start + limit)]

        result = donaciones.fetch_latest("losherederosdealberdi", 60, fetcher=fetcher)
        self.assertEqual(calls, [("losherederosdealberdi", 0, 50), ("losherederosdealberdi", 1, 50)])
        self.assertEqual(len(result), 60)
        self.assertEqual(result[0]["monto"], "$ 100")
        self.assertEqual(result[-1]["id"], 59)

    def test_stops_when_a_page_is_short(self):
        def fetcher(user, page, limit):
            if page == 0:
                return [{"id": 1, "valor": "10", "nombre": "Ana", "mensaje": "hola &amp; chau", "fecha": "Hace 1 hora", "privado": 1}]
            return []

        result = donaciones.fetch_latest("alguien", 10, fetcher=fetcher)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["nombre"], "Ana")
        self.assertEqual(result[0]["mensaje"], "hola & chau")
        self.assertTrue(result[0]["privada"])

    def test_none_count_walks_every_page_until_the_list_ends(self):
        calls = []

        def fetcher(user, page, limit):
            calls.append((page, limit))
            if page == 0:
                return [{"id": i, "valor": "10", "nombre": "n", "mensaje": "", "fecha": "", "privado": 0} for i in range(50)]
            if page == 1:
                return [{"id": 50, "valor": "10", "nombre": "ultima", "mensaje": "", "fecha": "", "privado": 0}, {"id": 1, "valor": "10", "nombre": "repetida", "mensaje": "", "fecha": "", "privado": 0}]
            return [{"id": 999, "valor": "1", "nombre": "no", "mensaje": "", "fecha": "", "privado": 0}]

        result = donaciones.fetch_latest("losherederosdealberdi", None, fetcher=fetcher)
        self.assertEqual(calls, [(0, 50), (1, 50)])
        self.assertEqual([item["id"] for item in result], list(range(50)) + [50])


class FilterAndFechaTests(unittest.TestCase):
    def test_parses_relative_dates(self):
        self.assertEqual(donaciones.fecha_age_seconds("Hace 7 horas"), 7 * 3600)
        self.assertEqual(donaciones.fecha_age_seconds("Hace 1 día"), 86400)
        self.assertEqual(donaciones.fecha_bucket("Hace 3 horas"), "Hoy")
        self.assertEqual(donaciones.fecha_bucket("Hace 2 semanas"), "Este mes")
        self.assertTrue(donaciones.matches_fecha("Hace 3 horas", "hoy"))
        self.assertFalse(donaciones.matches_fecha("Hace 2 meses", "semana"))
        self.assertTrue(donaciones.matches_fecha("Hace 7 horas", "7 horas"))

    def test_filters_by_user_date_and_amount(self):
        items = [
            {"nombre": "Ana", "fecha": "Hace 2 horas", "valor": "100", "monto": "$ 100", "mensaje": "", "privada": False},
            {"nombre": "Bruno", "fecha": "Hace 2 meses", "valor": "8000", "monto": "$ 8 000", "mensaje": "hola", "privada": False},
            {"nombre": "AnaPlus", "fecha": "Hace 1 día", "valor": "500", "monto": "$ 500", "mensaje": "x", "privada": True},
        ]
        only_ana = donaciones.filter_donations(items, donante="ana")
        self.assertEqual([item["nombre"] for item in only_ana], ["Ana", "AnaPlus"])
        this_week = donaciones.filter_donations(items, fecha="semana")
        self.assertEqual([item["nombre"] for item in this_week], ["Ana", "AnaPlus"])
        big = donaciones.filter_donations(items, minimo=500, maximo=1000)
        self.assertEqual([item["nombre"] for item in big], ["AnaPlus"])


class AnalyzeTests(unittest.TestCase):
    def test_totals_and_rankings(self):
        items = [
            {"nombre": "Ana", "fecha": "Hace 2 horas", "valor": "100", "monto": "$ 100", "mensaje": "https://x.com/a", "privada": False},
            {"nombre": "Ana", "fecha": "Hace 3 horas", "valor": "300", "monto": "$ 300", "mensaje": "", "privada": False},
            {"nombre": "Bruno", "fecha": "Hace 2 meses", "valor": "1000", "monto": "$ 1 000", "mensaje": "hola", "privada": True},
        ]
        report = donaciones.analyze(items, top=5)
        self.assertEqual(report["cantidad"], 3)
        self.assertEqual(report["suma"], 1400)
        self.assertEqual(report["donantes"], 2)
        self.assertEqual(report["recurrentes"], 1)
        self.assertEqual(report["privadas"], 1)
        self.assertEqual(report["con_link"], 1)
        self.assertEqual(report["por_usuario_monto"][0]["nombre"], "Bruno")
        self.assertEqual(report["por_usuario_cantidad"][0]["nombre"], "Ana")
        self.assertAlmostEqual(donaciones.gini_coefficient([1, 1, 1, 1]), 0.0)
        self.assertGreater(report["concentracion"]["gini_donantes"], 0)
        self.assertIn("p50", report["ciencia"]["percentiles"])
        text = donaciones.render_metricas(report, "losherederosdealberdi", "usuario=Ana")
        self.assertIn("Análisis de donaciones", text)
        self.assertIn("Top donantes por monto", text)
        self.assertIn("filtro: usuario=Ana", text)
        self.assertIn("Data science", text)
        self.assertIn("Lectura", text)

    def test_load_donations_reads_gzip(self):
        import gzip
        import json
        import tempfile
        from pathlib import Path

        payload = [{"id": 9, "valor": "10", "nombre": "Ana", "fecha": "Hace 1 hora", "mensaje": "", "privado": 0}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "donaciones.json.gz"
            path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
            loaded = donaciones.load_donations(str(path))
        self.assertEqual(loaded[0]["nombre"], "Ana")
        self.assertEqual(loaded[0]["monto"], "$ 10")


class RenderTests(unittest.TestCase):
    def test_text_includes_amount_name_and_message(self):
        text = donaciones.render_text(
            [
                {
                    "monto": "$ 500",
                    "nombre": "Disociandri",
                    "fecha": "Hace 7 horas",
                    "mensaje": "Hola",
                    "privada": False,
                }
            ],
            "losherederosdealberdi",
        )
        self.assertIn("Últimas 1 donaciones", text)
        todas = donaciones.render_text(
            [{"monto": "$ 1", "nombre": "Ana", "fecha": "", "mensaje": "", "privada": False}],
            "losherederosdealberdi",
            todas=True,
        )
        self.assertIn("Todas las donaciones (1)", todas)
        self.assertIn("https://ceneka.net/losherederosdealberdi", text)
        self.assertIn("$ 500", text)
        self.assertIn("Disociandri", text)
        self.assertIn("Hola", text)


if __name__ == "__main__":
    unittest.main()
