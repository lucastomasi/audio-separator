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
