"""
Quorum's structured world model: what the literature agent read, as linked findings,
plus every claim, its evidence and its verdict, in one SQLite file that lasts between runs.

- models.py   the tables (Tortoise ORM)
- schemas.py  the shapes going in and out (Pydantic)
- queries.py  every read and write (async), including the prefetch_related() run export
- store.py    WorldModel: the same, for the synchronous pipeline
- api.py      the same, over HTTP (FastAPI)
"""

from quorum.world.store import WorldModel

__all__ = ["WorldModel"]
