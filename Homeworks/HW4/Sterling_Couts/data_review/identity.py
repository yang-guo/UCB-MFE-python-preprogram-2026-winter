"""Apply only documented, date-bounded identity links; never infer from returns."""

import numpy as np
import pandas as pd


def attach_identities(frame, registry, date_column, *, table):
    """Require the source table: a ticker's price and financial issuers can differ."""
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])
    result["security_id"] = "unverified:" + result.symbol.astype(str)
    result["issuer_id"] = result["security_id"]
    result["identity_status"] = "unverified"
    result["source_priority"] = 999
    for item in registry:
        if table not in item["tables"]:
            continue
        for priority, symbol in enumerate(item["source_priority"]):
            mask = result.symbol.eq(symbol) & dates.between(
                item["scope_start"], item["scope_end"]
            )
            if result.loc[mask, "identity_status"].ne("unverified").any():
                raise ValueError("Overlapping identity rules require investigation.")
            result.loc[mask, "security_id"] = item["security_id"]
            result.loc[mask, "issuer_id"] = item["issuer_id"]
            result.loc[mask, "identity_status"] = "documented"
            result.loc[mask, "source_priority"] = priority
    return result


def collapse_identical_aliases(frame, key_columns, value_columns):
    """Collapse supported aliases only when requested values agree.

    No ranking by return, future coverage, current active status or number of
    missing fields. A documented, fixed source priority chooses equivalent rows.
    Conflicting rows remain in a separate review table and are not certified.
    """
    chosen, conflicts, removed = [], [], []
    for _, group in frame.groupby(key_columns, dropna=False, sort=False):
        group = group.sort_values(["source_priority", "symbol"], kind="stable")
        if len(group) == 1:
            chosen.append(group.iloc[0])
            continue
        if not group.identity_status.eq("documented").all():
            raise ValueError(
                "Duplicate unverified security keys are not safe to collapse."
            )
        first = group.iloc[0]
        matches = True
        for column in value_columns:
            a = group[column]
            if pd.api.types.is_numeric_dtype(a):
                matches &= bool(
                    np.isclose(
                        a.to_numpy(dtype=float),
                        float(first[column]),
                        rtol=1e-12,
                        atol=1e-9,
                        equal_nan=True,
                    ).all()
                )
            else:
                matches &= bool(
                    (a.eq(first[column]) | (a.isna() & pd.isna(first[column]))).all()
                )
        if matches:
            chosen.append(first)
            removed.extend(group.iloc[1:].to_dict("records"))
        else:
            conflicts.extend(group.to_dict("records"))
    # Preserve schema even when every alias conflicts or no duplicate is found.
    return tuple(
        pd.DataFrame(rows, columns=frame.columns)
        for rows in (chosen, removed, conflicts)
    )
