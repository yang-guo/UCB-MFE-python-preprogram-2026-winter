"""Simulate a lost acknowledgement after a local write has committed."""

import json
import sqlite3


def naive_write(db, note):
    with db:
        db.execute("INSERT INTO naive_notes(body) VALUES(?)", (note,))
    raise TimeoutError("The reply was lost after the commit")


def save_once(db, operation_key, note, *, lose_reply=False):
    """The effect and its receipt share a transaction and a stable business key."""
    with db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute("SELECT body,receipt FROM operations WHERE operation_key=?", (operation_key,)).fetchone()
        if previous:
            if previous[0] != note:
                raise ValueError("Operation key was reused with different arguments")
            receipt = json.loads(previous[1])
        else:
            row = db.execute("INSERT INTO checked_notes(body) VALUES(?)", (note,))
            receipt = {"note_id": row.lastrowid}
            db.execute("INSERT INTO operations VALUES(?,?,?)", (operation_key, note, json.dumps(receipt)))
    if lose_reply:
        raise TimeoutError("The reply was lost after the commit")
    return receipt


def experiment():
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
            CREATE TABLE naive_notes(id INTEGER PRIMARY KEY, body TEXT);
            CREATE TABLE checked_notes(id INTEGER PRIMARY KEY, body TEXT);
            CREATE TABLE operations(operation_key TEXT PRIMARY KEY, body TEXT, receipt TEXT);
        """)
        for _ in range(2):
            try:
                naive_write(db, "Investigate AAPL's debt maturities.")
            except TimeoutError:
                pass
        key = "project-17:AAPL:debt-question:v1"
        try:
            save_once(db, key, "Investigate AAPL's debt maturities.", lose_reply=True)
        except TimeoutError:
            pass
        receipt = save_once(db, key, "Investigate AAPL's debt maturities.")
        naive = db.execute("SELECT count(*) FROM naive_notes").fetchone()[0]
        checked = db.execute("SELECT count(*) FROM checked_notes").fetchone()[0]
        try:
            save_once(db, key, "An unrelated note with the same key.")
        except ValueError:
            pass
        else:
            raise AssertionError("A key must identify one operation and its arguments")
        return {"naive_writes": naive, "idempotent_writes": checked, "retry_receipt": receipt}


if __name__ == "__main__":
    result = experiment()
    print(json.dumps(result, indent=2))
    assert result["naive_writes"] == 2 and result["idempotent_writes"] == 1
