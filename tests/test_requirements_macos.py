"""Guard the clone-time pip set against known ResolutionImpossible pins."""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pins(text: str) -> dict[str, str]:
    out = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", line)
        if match:
            out[match.group(1).lower()] = match.group(2).strip()
    return out


class RequirementsMacosTests(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "requirements-macos.txt").read_text(encoding="utf-8")
        self.pins = _pins(self.text)

    def test_gradio_and_transformers_share_hub_major(self):
        # gradio 6.20 needs huggingface-hub>=1.2; transformers 4.x needs <1.
        self.assertEqual(self.pins.get("gradio"), "==6.20.0")
        spec = self.pins.get("transformers", "")
        self.assertTrue(
            spec.startswith("==5.") or spec.startswith(">=5"),
            f"transformers must be 5.x for gradio 6.20, got {spec!r}",
        )

    def test_no_infer_rvc_python(self):
        # infer-rvc-python pins torchcrepe→librosa==0.9.1; conversion uses .venv-vc.
        self.assertNotIn("infer-rvc-python", self.pins)

    def test_librosa_allows_coqui(self):
        # coqui-tts 0.27.5 needs librosa>=0.11.0
        spec = self.pins.get("librosa", "")
        self.assertIn("0.11", spec)
        self.assertNotIn("<0.11", spec)


if __name__ == "__main__":
    unittest.main()
