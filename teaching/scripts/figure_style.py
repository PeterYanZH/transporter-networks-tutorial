"""Explicit publication typography; no font files are redistributed.

Teaching supplement by PeterYanZH, derived from Transporter Networks (Zeng
et al., 2020), https://github.com/google-research/ravens. Not an official release.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
from pathlib import Path


def add_font_arguments(parser):
    parser.add_argument('--font-family', default='Times New Roman',
                        help='Exact installed font family; explicitly select e.g. Liberation Serif if needed')
    parser.add_argument('--font-dir', type=Path,
                        help='Optional directory of your own authorized TTF/OTF fonts (not distributed)')


def apply_style(root=None, *, font_family='Times New Roman', font_dir=None):
    """Select the requested font or fail; never silently substitute a font.

    root is accepted for compatibility. It is not searched for bundled fonts.
    Matplotlib discovers system fonts normally.
    """
    import matplotlib as mpl
    from matplotlib import font_manager
    if font_dir is not None:
        font_dir = Path(font_dir).expanduser()
        if not font_dir.is_dir():
            raise SystemExit(f'Font directory does not exist: {font_dir}')
        for font in sorted(font_dir.iterdir()):
            if font.suffix.lower() in {'.ttf', '.otf'}:
                font_manager.fontManager.addfont(str(font))
    try:
        font_manager.findfont(font_family, fallback_to_default=False)
    except ValueError as exc:
        raise SystemExit(
            f'Requested font {font_family!r} is unavailable. Install it with a '
            'valid license, pass --font-dir with your authorized font files, '
            'or explicitly choose an installed alternative such as '
            '--font-family "Liberation Serif". No font was substituted.'
        ) from exc
    mpl.rcParams.update({
        'font.family':font_family, 'font.serif':[font_family],
        'mathtext.fontset':'custom', 'mathtext.rm':font_family,
        'mathtext.it':f'{font_family}:italic', 'mathtext.bf':f'{font_family}:bold',
        'mathtext.sf':font_family, 'mathtext.tt':font_family,
        'mathtext.cal':font_family, 'font.cursive':[font_family],
        'mathtext.fallback':None,
        'pdf.fonttype':42, 'ps.fonttype':42, 'axes.unicode_minus':False,
        'font.size':10, 'axes.titlesize':10, 'axes.labelsize':10,
        'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9,
    })
