"""Tests for setup.py on a card with less VRAM than the sizes of strata-poly were measured on - a 6 GB NVIDIA
card (a GTX 1660 SUPER, 6,144 MiB, compute capability 7.5), where no size fits by the 32K figures.  Named for the
card: test_setup_small_card.py covers the ~8 GB cards, which fit a size and keep the RAM rule.  What that path
does: judge at the shortest context offered, because the KV cache is what the card is short of, and take the size
that puts the most experts in its VRAM.  Every other card keeps the 32K figures and the RAM rule, which the first
test proves by value.  Mocked - no GPU, no compile, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_small_card_6gb
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# what setup.gpus() reads for a 6,144 MiB card: MiB / 1024
SIX = 6144 / 1024.0
SMALL = ("qwen36", "gemma4")                   # the families whose smallest size fits a 6 GB card at 8K
Q2, Q3, Q4 = "UD-Q2_K_XL", "UD-Q3_K_M", "UD-Q4_K_M"


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Figures(unittest.TestCase):
    """The estimates, and what a context does to them."""

    def test_the_figures_are_the_ones_measured(self):
        """Without a context every figure is the one from an RTX 5090 at 32K, unchanged: this is what every card
        that fits a size sees, and what the numbers in docs/BACKENDS.md were computed from."""
        d = setup.MODELS["3.0BIT"]                                     # GLM 5.3 Flash, the measured one
        self.assertEqual(setup.poly_beside_gb(d), 12)
        self.assertAlmostEqual(setup.poly_held_gb(d, 31.84) / 1.073741824, 19.9, delta=0.2)
        self.assertAlmostEqual(setup.poly_ram_need(d, 31.84), 96, delta=1)
        for fam in setup.FAMILIES.values():                           # and for every poly size
            for m, s in fam.get("models", {}).items():
                self.assertEqual(setup.poly_beside_gb(s), s.get("vram_beside_gb", setup.GLM_VRAM_BESIDE_GB), fam["title"])
                self.assertEqual(setup.poly_beside_gb(s, None, 28672), setup.poly_beside_gb(s), fam["title"])
                self.assertEqual(setup.poly_beside_gb(s, setup.POLY_BESIDE_CTX, 28672), setup.poly_beside_gb(s),
                                 f"{fam['title']} {m}")

    def test_a_shorter_context_frees_its_kv_cache(self):
        """The KV cache is row_bytes a token, so a size takes less VRAM before any expert at 8K than at 32K - by
        that much, and nothing else changes."""
        for fam in SMALL:
            row = setup.FAMILIES[fam]["row_bytes"]
            for m, d in setup.family_models(fam).items():
                want = setup.poly_beside_gb(d) - (setup.POLY_BESIDE_CTX - 8192) * row / 2 ** 30
                self.assertAlmostEqual(setup.poly_beside_gb(d, 8192, row), want, delta=1e-9, msg=f"{fam} {m}")
                self.assertEqual(setup.poly_held_gb(d, SIX), setup.poly_held_gb(d, SIX, None, None))

    def test_a_longer_context_costs_more(self):
        """The other side of the same line, which nothing had covered: above the 32K the figure was measured at,
        the KV cache grows and takes it from the expert cache.  It used to stop there and report the 32K figure for
        every longer context, which on a small card is the figure that decides whether a size fits."""
        for fam in SMALL:
            row = setup.FAMILIES[fam]["row_bytes"]
            for m, d in setup.family_models(fam).items():
                base = setup.poly_beside_gb(d)
                self.assertAlmostEqual(setup.poly_beside_gb(d, setup.POLY_BESIDE_CTX, row), base, delta=1e-9)
                for c in (65536, 131072):
                    want = base + (c - setup.POLY_BESIDE_CTX) * row / 2 ** 30
                    self.assertAlmostEqual(setup.poly_beside_gb(d, c, row), want, delta=1e-9, msg=f"{fam} {m} {c}")
                    self.assertGreater(setup.poly_beside_gb(d, c, row), base, f"{fam} {m} {c}")
                    self.assertLess(setup.poly_held_gb(d, SIX, c, row), setup.poly_held_gb(d, SIX, 8192, row))

    def test_the_eighty_thousand_a_card_has_no_room_for(self):
        """Gemma 4 at 80,000 on this card: the KV cache alone asks for more than the card has beside the weights.
        The estimate leaves a sliver of the arena, under 2%, which is nothing to answer with.  These are the
        figures carried by arithmetic from the measured 16,384 run, not a run at that context."""
        row = setup.FAMILIES["gemma4"]["row_bytes"]
        d = setup.family_models("gemma4")[Q2]
        self.assertGreater(80000 * row / 2 ** 30, 1.1)          # more KV than the 1.1 GiB of experts measured
        self.assertFalse(setup.poly_fits(d, 32.0, SIX, False, 80000, row))
        self.assertLess(setup.poly_held_gb(d, SIX, 80000, row) / d["arena_gb"], 0.02)

    def test_the_six_gb_card_is_under_every_size_at_32k(self):
        """What makes this a small card: not one size fits by the figure it was measured at.  Ornith is not - its
        APEX-I-MINI takes 3.9 GB and so fits - and must keep the RAM rule."""
        for fam in SMALL:
            for m, d in setup.family_models(fam).items():
                self.assertTrue(setup.poly_too_small(d, SIX), f"{fam} {m} should be under a 6 GB card at 32K")
        ornith = setup.family_models("ornith")
        self.assertFalse(setup.poly_too_small(ornith["APEX-I-MINI"], SIX))
        self.assertEqual(setup.poly_default(ornith, 32.0, SIX), "APEX-I-COMPACT")

    def test_the_smallest_size_fits_it_at_8k(self):
        for fam in SMALL:
            row = setup.FAMILIES[fam]["row_bytes"]
            self.assertTrue(setup.poly_fits(setup.family_models(fam)[Q2], 32.0, SIX, False, 8192, row),
                            f"{fam} {Q2} should fit a 6 GB card at 8K")
            self.assertFalse(setup.poly_fits(setup.family_models(fam)[Q2], 32.0, SIX, False))

    def test_the_card_holds_a_quarter_of_the_experts(self):
        """The share the sizes below put on the card, for the record; these are estimates, not measurements."""
        for fam in SMALL:
            row = setup.FAMILIES[fam]["row_bytes"]
            d = setup.family_models(fam)[Q2]
            share = setup.poly_held_gb(d, SIX, 8192, row) / d["arena_gb"]
            self.assertGreater(share, 0.2, fam)
            self.assertLess(share, 0.3, fam)

    def test_the_size_that_fits_is_the_one_taken(self):
        """The RAM rule would take the largest file, which is the one that leaves the card least room."""
        for fam in SMALL:
            sizes = setup.family_models(fam)
            row = setup.FAMILIES[fam]["row_bytes"]
            self.assertEqual(setup.poly_default(sizes, 32.0, SIX, 8192, row), Q2, fam)
            # and it is the one that holds the most
            self.assertEqual(max(sizes, key=lambda m: setup.poly_held_gb(sizes[m], SIX, 8192, row)), Q2, fam)

    def test_the_eight_gb_laptop_keeps_the_ram_rule(self):
        """8 GB is a card these sizes have run on (docs/BACKENDS.md).  Nothing here may change for it."""
        for fam in SMALL:
            sizes = setup.family_models(fam)
            self.assertEqual(setup.poly_default(sizes, 32.0, 8.0), Q3, fam)
            self.assertFalse(setup.poly_check_ctx(setup.FAMILIES[fam], 8.0), fam)

    def test_the_check_judges_a_small_card_at_the_shortest_context(self):
        for fam in SMALL:
            self.assertEqual(setup.poly_check_ctx(setup.FAMILIES[fam], SIX), setup.POLY_CONTEXTS[0], fam)
        for vram in (8.0, 12.0, 16.0, 24.0, 31.84):                   # every card the project supports
            for fam in setup.FAMILIES:
                sizes = setup.family_models(fam)
                if not sizes or any(not setup.poly_too_small(d, vram) for d in sizes.values()):
                    # a card one of the family's sizes fits at 32K: the 32K figures are its figures
                    self.assertIsNone(setup.poly_check_ctx(setup.FAMILIES[fam], vram), f"{fam} at {vram}")

    def test_the_check_line_says_what_the_card_gets(self):
        """At 32K, the figure it was measured at, setup says the size does not fit and names the GB.  Judged at
        8K - which is what a small card is judged at - the same size says it fits, and how much RAM it wants."""
        row = setup.FAMILIES["gemma4"]["row_bytes"]
        d = setup.family_models("gemma4")[Q2]
        said = setup.poly_verdict(d, 32.0, SIX, False)
        self.assertIn("does not fit", said)
        self.assertIn("6.6 GB of VRAM", said)                           # 4.6 of weights and caches, 2 of room
        self.assertNotIn("at 8K", said)
        at8 = setup.poly_verdict(d, 32.0, SIX, False, True, 8192, row)
        self.assertIn("fits by setup's estimate", at8)
        self.assertIn("~16 GB of RAM", at8)
        self.assertIn("AMD", setup.poly_verdict(d, 32.0, SIX, True, False, 8192, row))

    def test_an_amd_card_is_not_offered_a_short_context(self):
        self.assertEqual(setup.poly_check_ctx(setup.FAMILIES["gemma4"], SIX), setup.POLY_CONTEXTS[0])
        self.assertIn("does not run on AMD cards",
                      setup.poly_verdict(setup.family_models("gemma4")[Q2], 32.0, SIX, True))


class Install(unittest.TestCase):
    """A whole install on a 6 GB card, mocked."""

    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.runs, self.downloads, self.verified, self.built = [], [], [], []

    def main(self, family="gemma4", model=Q2, ram=32.0, vram=SIX, argv=()):
        found = [{"index": 0, "name": "NVIDIA GeForce GTX 1660 SUPER", "vram_gb": vram, "arch": "75",
                  "driver": "580.97"}]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_build(gpu, yes, llama, vision="none", cuda_tk=None):
            self.built.append(gpu["archs"])
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "cuda_dirs": ["/cuda/lib64"]}))
            return eng

        with contextlib.ExitStack() as st:
            for p in [
                mock.patch.object(setup, "ROOT", self.t),
                mock.patch.object(setup, "GPU_PICK", None),
                mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
                mock.patch.object(setup, "installed_configs", lambda: []),
                mock.patch.object(setup, "gpus", lambda: found),
                mock.patch.object(setup, "amd_gpus", lambda *a: []),
                mock.patch.object(setup, "ram_gb", lambda: ram),
                mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
                mock.patch.object(setup, "cpu_cores", lambda: None),
                mock.patch.object(setup, "page_file_gb", lambda: 16.0),
                mock.patch.object(setup, "free_gb", lambda p: 500.0),
                mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
                mock.patch.object(setup, "build_engine_poly", fake_build),
                mock.patch.object(setup, "download", fake_download),
                mock.patch.object(setup, "whole_shard", lambda s: False),
                mock.patch.object(setup, "check_shards", lambda shards: None),
                mock.patch.object(setup, "verify_sha256",
                                  lambda s, size, sha: self.verified.append((s.name, size, sha))),
                mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
                mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
                mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
                mock.patch.object(sys, "argv", ["setup.py", "--family", family, "--model", model, "--yes",
                                                "--no-start", "--models-dir", str(self.t / "models"), *argv]),
            ]:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = next(iter(sorted(self.t.glob(f"strata-{setup.FAMILIES[family]['tag']}*.json"))), None)
        return code, out, (json.loads(cfg.read_text()) if cfg else None)

    def test_the_small_size_and_the_short_context_are_written(self):
        code, out, cfg = self.main()
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(Path(args[args.index("--model") + 1]).name,
                         setup.model_file(setup.FAMILIES["gemma4"], Q2, 1))
        self.assertNotIn("vision", cfg)                                # no image encoder on a card this small

    def test_the_card_is_not_told_it_cannot_run(self):
        _, out, _ = self.main()
        self.assertNotIn("going on with 6 GB of VRAM", out)
        self.assertIn("8K", out)
        self.assertEqual(self.built, [["75"]])                          # compiled for Turing, as asked

    def test_qwen36_the_same(self):
        code, out, cfg = self.main(family="qwen36")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "8192")

    def test_a_named_context_is_used_as_it_is(self):
        _, _, cfg = self.main(argv=("--context", "32768"))
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "32768")

    def test_a_named_context_is_judged_by(self):
        """#the bug this path had: a --context of 8192 was written to the config but the sizing above it was still
        done at 32K, so setup refused a card it was about to fit (it asks to go on anyway).  With the context
        named, the figures are that context's; at 32K on 6 GB they are the ones that do not fit."""
        _, ok8, _ = self.main(argv=("--context", "8192"))
        self.assertNotIn("going on with 6 GB of VRAM", ok8)              # it fits: no risk asked
        self.assertIn("about 25% of Gemma 4 26B-A4B's experts fit in VRAM", ok8)
        _, ok32, _ = self.main(argv=("--context", "32768"))
        self.assertIn("takes about 4.6 GB of VRAM before any expert", ok32)
        self.assertIn("going on with 6 GB of VRAM, as you chose", ok32)   # 32K does not fit, --yes takes the risk

    def test_a_longer_context_is_said_to_cost_more(self):
        """Above the 32K the figure was measured at, the KV cache grows: 65,536 asks for 0.66 GB more than 32K
        does, and setup says so rather than quoting the 32K figure again."""
        _, out32, _ = self.main(argv=("--context", "32768"))
        _, out65, _ = self.main(argv=("--context", "65536"))

        def said(out):                       # the "takes about N GB of VRAM before any expert" a risk prints
            line = next(ln for ln in out.splitlines() if "takes about" in ln)
            return float(line.split("about ")[1].split(" GB")[0])

        row = setup.FAMILIES["gemma4"]["row_bytes"]
        self.assertAlmostEqual(said(out32), setup.poly_beside_gb(setup.family_models("gemma4")[Q2], 32768, row),
                               delta=0.05)
        self.assertAlmostEqual(said(out65) - said(out32), (65536 - 32768) * row / 2 ** 30, delta=0.05)
        self.assertGreater(said(out65), said(out32))

    def test_tail_skip_is_off_unless_asked(self):
        """qwen36 carries 6 of its own (a measured value); nothing is added to another family silently, and the
        flag the user can set is written when they do."""
        _, _, cfg = self.main(family="qwen36")
        self.assertEqual(cfg["args"][cfg["args"].index("--tail-skip") + 1], "6")
        _, _, cfg = self.main()
        self.assertNotIn("--tail-skip", cfg["args"])
        _, out, cfg = self.main(argv=("--tail-skip", "4"))
        self.assertEqual(cfg["args"][cfg["args"].index("--tail-skip") + 1], "4")
        self.assertIn("the answers move a little", out)
        _, out, cfg = self.main(argv=("--tail-skip", "0"))
        self.assertNotIn("--tail-skip", cfg["args"])                   # 0 turns it off, even for qwen36
        self.assertIn("this one has not", out)                        # and setup says it dropped it


if __name__ == "__main__":
    unittest.main()