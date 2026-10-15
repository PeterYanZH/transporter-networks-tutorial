#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
"""Check the synthetic teaching examples without training or plotting.

Teaching supplement derived from Transporter Networks (Zeng et al., 2020),
https://github.com/google-research/ravens. Requires NumPy only.
Outputs are generated in a temporary directory; bundled evidence is preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    import numpy as np

    root = args.root.expanduser().resolve()
    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='transporter-numerics-') as temporary:
        output = Path(temporary)
        subprocess.run([
            sys.executable, str(scripts/'toy_experiments.py'),
            '--out', str(output/'toys'), '--no-plots',
        ], check=True, stdout=subprocess.DEVNULL)
        old = json.loads((root/'toys/toy_metrics.json').read_text(encoding='utf-8'))
        new = json.loads((output/'toys/toy_metrics.json').read_text(encoding='utf-8'))
        # Equal-height modes can change their argsort order across NumPy builds.
        assert set(map(tuple, old['multimodal']['top_four_xy'])) == set(
            map(tuple, new['multimodal']['top_four_xy']))
        names = ['toy_correlation_raw.npz', 'toy_multimodal_raw.npz', 'toy_gradient_raw.npz']
        for name in names:
            with np.load(root/'toys'/name, allow_pickle=False) as expected, np.load(
                output/'toys'/name, allow_pickle=False
            ) as actual:
                assert expected.files == actual.files
                for key in expected.files:
                    # Central finite differences amplify roundoff by 1/epsilon.
                    tolerance = 1e-8 if key == 'finite_difference_gradient' else 1e-12
                    np.testing.assert_allclose(expected[key], actual[key],
                                               rtol=tolerance, atol=tolerance)
        subprocess.run([
            sys.executable, str(scripts/'plot_sliding_correlation.py'),
            '--root', str(output), '--no-plots',
        ], check=True, stdout=subprocess.DEVNULL)
        sliding = json.loads((output/'figures/sliding_correlation.json').read_text(encoding='utf-8'))
        assert sliding['scores'] == [[6, 17, 13], [17, 9, 11], [11, 10, 7]]
        assert sliding['maximum_locations_rc'] == [[0, 1], [1, 0]]
        assert sliding['independent_einsum_check_passed']
        print(json.dumps({
            'synthetic_checks_passed': True,
            'correlation_translation_error': new['correlation']['translation_overlap_max_absolute_error'],
            'multimodal_legal_probability_mass': new['multimodal']['legal_probability_mass'],
            'finite_difference_max_error': new['gradient']['gradient_check_max_absolute_error'],
            'three_channel_scores': sliding['scores'],
            'model_inferences': 0,
            'robot_training_updates': 0,
        }, indent=2))


if __name__ == '__main__':
    main()
