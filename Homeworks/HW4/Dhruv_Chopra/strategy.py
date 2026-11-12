"""Staggered-cohort, top-N strategy driven by the model's predicted 90-day rank.

Capital is split into three overlapping cohorts, each held for 90 calendar
days (~62 sessions). A new cohort is formed at most every 20 sessions, and
only when there is idle cash. So after the ramp-up (three cohorts ~1 month
apart), each maturing cohort's sale proceeds are reinvested at the next
session's close. This keeps the portfolio ~fully invested without ever
needing the same cash twice.
"""

import math

import numpy as np
import pandas as pd

from model import tradeable_mask


WEIGHTINGS = ("equal", "rank", "inverse_vol")


def cohort_weights(weighting, n_picked, vol=None):
    """Fractions of a cohort's budget for the picked names (best first).

    equal       1/n each.
    rank        linear in rank: best gets n, worst gets 1 (best ~2x average).
    inverse_vol proportional to 1 / 63-day volatility at the signal close.
    """
    if weighting == "equal":
        w = np.ones(n_picked)
    elif weighting == "rank":
        w = np.arange(n_picked, 0, -1, dtype=float)
    else:
        w = 1.0 / np.asarray(vol, dtype=float)
    return w / w.sum()


def capped_allocation(budget, weights, rooms, backup_rooms=(), backup_size=0.0):
    """Split `budget` by `weights` without exceeding each name's room (cap).

    Money a capped name cannot take is re-split over the uncapped picked names
    in proportion to their weights (repeated until nothing is capped). What is
    still left goes to the backup names (next in the ranking), up to
    `backup_size` each. Returns (amounts for picked, amounts for backups).
    """
    weights = np.asarray(weights, dtype=float)
    rooms = np.maximum(np.asarray(rooms, dtype=float), 0.0)
    amount = np.zeros_like(weights)
    remaining = float(budget)
    active = rooms > 1e-9
    while remaining > 1e-6 and active.any():
        share = remaining * weights[active] / weights[active].sum()
        add = np.minimum(share, rooms[active] - amount[active])
        amount[active] += add
        remaining -= add.sum()
        active &= (rooms - amount) > 1e-9
    backups = []
    for room in backup_rooms:
        give = min(backup_size, max(room, 0.0), remaining)
        backups.append(give)
        remaining -= give
    return amount, np.array(backups)


class CohortTopNStrategy:
    def __init__(self, data, feats, schedule, top_n=30, n_cohorts=3, min_gap_sessions=20,
                 min_cash_fraction=0.02, name_cap=0.05, sector_cap=0.25, weighting="equal",
                 redistribute=False, min_score=None, min_agreement=None, per_sector=None,
                 corr_cap=None, corr_window=252, corr_depth=300, corr_min_obs=200,
                 rotation=None):
        self.redistribute = redistribute     # re-split money a name cap would leave in cash
        # Signal cutoff: a top-N name whose predicted score (the model's predicted
        # 90-day return percentile) is below min_score is not bought; its slot
        # stays in cash. None = always fill every slot.
        self.min_score = min_score
        # Feature-agreement cutoff (see ModelSchedule.agreement); same cash rule.
        self.min_agreement = min_agreement
        # Sector-neutral selection: buy the per_sector best-scoring names in
        # *every* sector (top_n and sector_cap are then not used for picking).
        self.per_sector = per_sector
        # Correlation-limited greedy selection: walk down the ranking (at most
        # corr_depth names) and accept a stock only if the correlation of its
        # past `corr_window` daily returns with the current portfolio (held
        # names + names already accepted for this cohort, equal-weighted) is
        # below corr_cap. Unfilled slots stay in cash.
        self.corr_cap, self.corr_window = corr_cap, corr_window
        self.corr_depth, self.corr_min_obs = corr_depth, corr_min_obs
        if corr_cap is not None:
            px = data.adj_close
            ret = np.full_like(px, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                ret[1:] = px[1:] / px[:-1] - 1
            self._ret = ret
        self._corr_stats = {}
        # Industry rotation: dict(n_industries=1, lookback=63, per_industry=10,
        # within="score" | "momentum", min_members=10). Rank industries by the
        # equal-weighted past-`lookback`-session return of their tradeable
        # stocks; buy `per_industry` stocks of each of the top n_industries,
        # chosen by model score or by their own past return.
        self.rotation = rotation
        if rotation is not None:
            self.industries = data.industry.reindex(data.symbols).fillna("Unknown").to_numpy()
        self._rotation_stats = {}
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        self.weighting = weighting
        self.data, self.feats, self.schedule = data, feats, schedule
        self.top_n, self.n_cohorts, self.min_gap = top_n, n_cohorts, min_gap_sessions
        self.min_cash_fraction, self.name_cap, self.sector_cap = min_cash_fraction, name_cap, sector_cap
        self.sectors = data.sector.reindex(data.symbols).fillna("Unknown").to_numpy()
        self.last_cohort_t = None
        self.log = []                              # one row per cohort decision

    def _sector_neutral(self, cols, order):
        """Best `per_sector` names of each sector, listed in overall score
        order. Backups (for cap redistribution) are each sector's next names."""
        taken, picked, backup = {}, [], []
        for i in order:
            sector = self.sectors[cols[i]]
            k = taken.get(sector, 0)
            if k < self.per_sector:
                picked.append(cols[i])
            elif k < 2 * self.per_sector and self.redistribute:
                backup.append(cols[i])
            taken[sector] = k + 1
        return picked, backup

    def _rotation_select(self, cols, scores, t):
        """Industry rotation pick (see __init__). Uses prices through t only."""
        r = self.rotation
        L = r.get("lookback", 63)
        if t < L:
            return []
        px = self.data.adj_close
        with np.errstate(invalid="ignore", divide="ignore"):
            past = px[t, cols] / px[t - L, cols] - 1
        ind = self.industries[cols]
        frame = pd.DataFrame({"col": cols, "score": scores, "past": past, "ind": ind}).dropna(subset=["past"])
        size = frame.groupby("ind").col.transform("size")
        frame = frame[size >= r.get("min_members", 10)]
        if frame.empty:
            return []
        ind_ret = frame.groupby("ind").past.mean().sort_values(ascending=False)
        chosen = list(ind_ret.index[: r.get("n_industries", 1)])
        key = "score" if r.get("within", "score") == "score" else "past"
        picked = []
        for name in chosen:
            top = frame[frame.ind == name].sort_values([key, "col"], ascending=[False, True])
            picked += top.col.head(r.get("per_industry", 10)).tolist()
        self._rotation_stats = {"industries": "|".join(chosen),
                                "industry_return": float(ind_ret.iloc[0])}
        # Overall score order, so rank weights favour the model's best picks.
        return [c for c in frame.sort_values("score", ascending=False).col if c in set(picked)]

    def _corr_select(self, cols, order, t, snap):
        """Greedy selection with a cap on correlation to the current portfolio."""
        R = self._ret[max(0, t - self.corr_window + 1): t + 1]      # returns through the signal close
        total, count = np.zeros(len(R)), np.zeros(len(R))

        def add(n):
            r = R[:, n]
            ok = np.isfinite(r)
            total[ok] += r[ok]
            count[ok] += 1

        for n, value in snap.holdings.items():
            if value > 0:
                add(int(n))
        max_per_sector = math.ceil(self.sector_cap * self.top_n)
        picked, per_sector, rhos, depth = [], {}, [], 0
        for depth, i in enumerate(order[: self.corr_depth], start=1):
            n, sector = cols[i], self.sectors[cols[i]]
            if per_sector.get(sector, 0) >= max_per_sector:
                continue
            if count.any():
                port = np.where(count > 0, total / np.maximum(count, 1), np.nan)
                r = R[:, n]
                ok = np.isfinite(r) & np.isfinite(port)
                if ok.sum() < self.corr_min_obs:
                    continue
                rho = np.corrcoef(r[ok], port[ok])[0, 1]
                if not rho < self.corr_cap:
                    continue
                rhos.append(rho)
            picked.append(n)
            per_sector[sector] = per_sector.get(sector, 0) + 1
            add(n)
            if len(picked) == self.top_n:
                break
        self._corr_stats = {"mean_corr_of_picks": float(np.mean(rhos)) if rhos else np.nan,
                            "search_depth": depth}
        return picked

    def plan(self, t, snap):
        """Return {column index: fee-inclusive budget} for a fill at the next close."""
        if self.last_cohort_t is not None and t - self.last_cohort_t < self.min_gap:
            return {}
        if snap.nav <= 0 or snap.cash < self.min_cash_fraction * snap.nav:
            return {}
        # Cohorts that are still running (not yet due) at the signal close.
        live = {lot.signal_date for lot in snap.open_lots if lot.due_date > snap.date}
        budget = snap.cash if len(live) >= self.n_cohorts - 1 else min(snap.cash, snap.nav / self.n_cohorts)

        cols, scores = self.schedule.scores(t)
        ok = tradeable_mask(self.data, self.feats, t)[cols]
        cols, scores = cols[ok], scores[ok]
        if not len(cols):
            return {}
        # Highest score first; ticker breaks ties deterministically.
        order = np.lexsort((self.data.symbols.to_numpy()[cols], -scores))
        if self.rotation is not None:
            picked, backup = self._rotation_select(cols, scores, t), []
        elif self.corr_cap is not None:
            picked, backup = self._corr_select(cols, order, t, snap), []
        elif self.per_sector is not None:
            picked, backup = self._sector_neutral(cols, order)
        else:
            max_per_sector = math.ceil(self.sector_cap * self.top_n)
            n_backup = self.top_n if self.redistribute else 0
            picked, backup, per_sector = [], [], {}
            for i in order:
                sector = self.sectors[cols[i]]
                if per_sector.get(sector, 0) >= max_per_sector:
                    continue
                (picked if len(picked) < self.top_n else backup).append(cols[i])
                per_sector[sector] = per_sector.get(sector, 0) + 1
                if len(picked) == self.top_n and len(backup) >= n_backup:
                    break

        vol = self.feats["vol_63"][t, picked] if self.weighting == "inverse_vol" else None
        weights = cohort_weights(self.weighting, len(picked), vol)
        score_of = dict(zip(cols.tolist(), scores.tolist()))
        n_full = len(picked)
        top_scores = [score_of[int(n)] for n in picked]
        agree_fn = getattr(self.schedule, "agreement", None)
        top_agreement = (agree_fn(t, np.array(picked + backup, dtype=int)).tolist()
                         if agree_fn is not None and picked else [np.nan] * (len(picked) + len(backup)))
        agree_of = dict(zip([int(n) for n in picked + backup], top_agreement))
        top_agreement = top_agreement[:len(picked)]
        top_picked = [int(n) for n in picked]

        def passes(n):
            ok_ = True
            if self.min_score is not None:
                ok_ &= score_of[int(n)] >= self.min_score
            if self.min_agreement is not None:
                ok_ &= bool(agree_of.get(int(n), np.nan) >= self.min_agreement)
            return ok_

        if self.min_score is not None or self.min_agreement is not None:
            # Slots that fail the cutoff keep their share of the budget in cash.
            passing = np.array([passes(n) for n in picked], dtype=bool)
            budget *= float(weights[passing].sum())
            picked = [n for n, ok_ in zip(picked, passing) if ok_]
            weights = weights[passing] / weights[passing].sum() if passing.any() else weights[:0]
            backup = [n for n in backup if passes(n)]
        # Single-name cap: total exposure after this buy <= name_cap * NAV.
        room = lambda n: self.name_cap * snap.nav - snap.holdings.get(n, 0.0)
        if not picked:
            amounts, backup_amounts = np.array([]), np.array([])
        elif self.redistribute:
            amounts, backup_amounts = capped_allocation(
                budget, weights, [room(n) for n in picked], [room(n) for n in backup],
                backup_size=budget / max(len(picked), 1))
        else:
            amounts = np.array([min(budget * w, max(room(n), 0.0)) for n, w in zip(picked, weights)])
            backup_amounts = np.array([])
        orders = {}
        for n, amount in list(zip(picked, amounts)) + list(zip(backup, backup_amounts)):
            if amount > 1.0:
                orders[int(n)] = float(amount)
        self.last_cohort_t = t
        self.log.append({"signal_date": snap.date, "cash": snap.cash, "nav": snap.nav, "budget": budget,
                         "names": len(orders), "invested": sum(orders.values()), "live_cohorts": len(live),
                         "candidates": len(cols), "slots_filled": len(picked) / max(n_full, 1),
                         "score_best": float(scores.max()), "top_scores": top_scores,
                         "top_agreement": top_agreement, "top_picked": top_picked,
                         **self._corr_stats, **self._rotation_stats,
                         "score_nth": float(np.sort(scores)[::-1][min(self.top_n, len(scores)) - 1])})
        return orders
