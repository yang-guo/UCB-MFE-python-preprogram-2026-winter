# Symbols removed from the investable universe, feature panels and labels.
# The source database is left untouched; every derived file applies this list.
#
# AAM: the ticker mixes unrelated securities. 2010-2022 rows trade at
# $7,000-$37,000 with zero-volume rows at $0.19-$0.61 interleaved, and the
# snapshot name is a 2024 SPAC. It also ranks #1 in universe_membership in
# 2011-13 from these prices, producing forward returns above +14,000,000%.
#
# VBIV: adjusted close jumps from $1.11 to $924 on 2010-04-26, a zero-volume
# session, with close equal to adj_close and no return to the old level. The
# pre-jump series looks stitched or unadjusted; it produced the 20 largest
# forward returns (about +66,000% to +84,000%).
#
# LCI: on 2013-12-13 the raw close jumps from $28.09 to $118.52 (4.2x) on
# lower volume, and adj_close is simultaneously set to 4x the close ($474.08;
# close/adj_close goes from 1.00 to 0.25), a 16.9x one-day adjusted move.
# Adjusted prices then stay near $650-$880 through 2014-15. These stacked
# adjustments are a data error; they produced the 20 largest forward returns
# after VBIV was removed (about +2,500% to +3,000%).
EXCLUDED_SYMBOLS = ("AAM", "VBIV", "LCI")


def excluded_symbols_sql():
    """Return the exclusion list as a SQL tuple literal, e.g. ('AAM')."""
    return "(" + ", ".join(f"'{s}'" for s in EXCLUDED_SYMBOLS) + ")"
