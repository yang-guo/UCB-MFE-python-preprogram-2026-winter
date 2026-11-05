"""Durable project memory. SQLite commits evidence and receipts together."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .data import canonical


def now():
    return datetime.now(timezone.utc).isoformat()


def connect(run):
    db = sqlite3.connect(Path(run) / "memory.sqlite", timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize(run, tickers):
    with connect(run) as db:
        db.executescript("""
            CREATE TABLE tasks(ticker TEXT PRIMARY KEY, status TEXT NOT NULL,
                               attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
            CREATE TABLE evidence(id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                                  ticker TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE notes(ticker TEXT PRIMARY KEY REFERENCES tasks(ticker),
                               body TEXT NOT NULL, citations TEXT NOT NULL);
            CREATE TABLE calls(id INTEGER PRIMARY KEY, ticker TEXT, attempt INTEGER,
                               tool TEXT, created_at TEXT);
            CREATE TABLE events(id INTEGER PRIMARY KEY, created_at TEXT, kind TEXT, payload TEXT);
        """)
        db.executemany("INSERT INTO tasks(ticker,status) VALUES(?,'pending')", [(t,) for t in tickers])


def event(db, kind, payload):
    db.execute("INSERT INTO events(created_at,kind,payload) VALUES(?,?,?)",
               (now(), kind, canonical(payload)))


def remember(db, kind, ticker, payload):
    record = evidence_record(kind, ticker, payload)
    db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)",
               (record["id"], kind, ticker, canonical(payload)))
    return record


def evidence_record(kind, ticker, payload):
    """Build an immutable record before deciding whether to store or retrieve it."""
    encoded = canonical(payload)
    identifier = hashlib.sha256(f"{kind}:{ticker}:{encoded}".encode()).hexdigest()[:20]
    return {"id": identifier, "kind": kind, "ticker": ticker, "payload": payload}


def evidence(db, ticker=None):
    rows = db.execute("SELECT * FROM evidence WHERE ticker=? ORDER BY kind,id", (ticker,)) if ticker else db.execute("SELECT * FROM evidence ORDER BY ticker,kind,id")
    return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]


@contextmanager
def run_lock(run):
    """The operating system releases this lock if the supervisor crashes."""
    stream = (Path(run) / ".run.lock").open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if __import__("os").name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another supervisor is using this project") from None
    try:
        yield
    finally:
        stream.close()
