"""The published macOS requirement files must be internally satisfiable."""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


class MacosRequirementsTests(unittest.TestCase):
    def test_app_venv_keeps_intel_torch_and_gradio(self):
        pins = _pins(ROOT / "requirements-macos.txt")
        self.assertIn("torch==2.2.2", pins)
        self.assertIn("torchaudio==2.2.2", pins)
        self.assertIn("gradio==6.20.0", pins)
        self.assertIn("scikit-learn", pins)
        self.assertTrue(any(p.startswith("huggingface-hub") for p in pins))

    def test_app_venv_does_not_pin_conflicting_ml_stack(self):
        pins = _pins(ROOT / "requirements-macos.txt")
        joined = "\n".join(pins)
        self.assertNotIn("transformers==", joined)
        self.assertNotIn("infer-rvc-python", joined)
        self.assertNotIn("coqui-tts", joined)

    def test_vc_venv_keeps_known_good_transformers(self):
        pins = _pins(ROOT / "requirements-vc.txt")
        self.assertIn("torch==2.2.2", pins)
        self.assertIn("torchaudio==2.2.2", pins)
        self.assertIn("transformers==4.53.3", pins)
        self.assertIn("huggingface-hub==0.36.2", pins)
        self.assertTrue(any(p.startswith("librosa") for p in pins))

    def test_space_pins_untouched(self):
        pins = _pins(ROOT / "requirements.txt")
        self.assertIn("torch==2.9.1", pins)
        self.assertIn("gradio==6.20.0", pins)


if __name__ == "__main__":
    unittest.main()
