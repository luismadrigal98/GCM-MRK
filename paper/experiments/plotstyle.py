"""Figure settings shared by every script that writes a manuscript figure.

Two settings matter for print rather than for a screen.

**Font type.** Matplotlib's default PDF backend embeds text as Type 3 fonts, a
bitmap-era construct that many publishers reject at submission and that some
viewers render poorly.  ``pdf.fonttype = 42`` embeds TrueType instead, so the
text stays selectable, searchable and scalable.

**Width.** A figure is placed in the manuscript at ``\\linewidth``.  A figure
drawn at :data:`FIG_WIDTH` is therefore reproduced at 1:1 and its labels appear
at the point size the script asked for; one drawn wider is scaled down and its
labels shrink with it.  A 7 pt tick label in a 7.2-inch figure lands at under
4 pt on the page, which is below what any journal accepts, so every figure in
the manuscript is drawn at this width.

Importing this module applies the settings; :data:`FIG_WIDTH` and
:data:`FIG_HEIGHT` are for the caller to pass to ``plt.subplots``.
"""

from __future__ import annotations

import matplotlib

#: Manuscript \textwidth (415.13 pt at 72.27 pt/inch), in inches.
FIG_WIDTH = 5.74

#: A comfortable default height for a single-panel figure at that width.
FIG_HEIGHT = 3.4

matplotlib.rcParams.update({
    "pdf.fonttype": 42,     # TrueType, not Type 3
    "ps.fonttype": 42,
})

__all__ = ["FIG_WIDTH", "FIG_HEIGHT"]
