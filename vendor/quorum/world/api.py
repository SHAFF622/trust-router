"""
The world model over HTTP (FastAPI). A coding agent, the web app or a person can
ask it what is known, without reading papers again.

    python -m quorum.world.api            -> http://127.0.0.1:8791/docs

Endpoints:
    GET /world/stats                       how much the world model holds
    GET /world/entities?kind=gene          subjects and outcomes, most-studied first
    GET /world/findings?subject=ESR1&outcome=DSS
    GET /world/graph?subject=ESR1          nodes and edges, for drawing
    GET /world/runs/{run_id}               a whole run: claims, evidence, verdicts, linked findings
    GET /world/claims/{claim_id}           one claim, the same way
    GET /world/brief?q=...                 what is known about the things a sentence names, as prompt-ready lines
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from tortoise.contrib.fastapi import RegisterTortoise

from quorum.world import queries
from quorum.world.schemas import ClaimOut, EntityOut, FindingOut, Graph, RunExport, Stats
from quorum.world.store import DEFAULT_PATH


def create_app(db_path: Optional[str] = None) -> FastAPI:
    path = db_path or os.environ.get("QUORUM_WORLD_DB") or str(DEFAULT_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Tortoise's own FastAPI hook: it opens the database for the app's whole life, so every
        # request handler sees the connection (a plain init here would only be visible at startup).
        async with RegisterTortoise(app, db_url=f"sqlite://{path}", modules=queries.MODULES, generate_schemas=True):
            yield

    app = FastAPI(title="Quorum world model", version="1",
                  description="What the literature agent has read, as structured findings, linked to the claims "
                              "the referees judged.", lifespan=lifespan)

    @app.get("/world/stats", response_model=Stats)
    async def stats():
        return await queries.stats()

    @app.get("/world/entities", response_model=list[EntityOut])
    async def entities(kind: Optional[str] = Query(None, description="gene, marker, subtype, treatment, clinical, "
                                                                     "other or outcome")):
        return await queries.entities(kind)

    @app.get("/world/findings", response_model=list[FindingOut])
    async def findings(subject: Optional[str] = None, outcome: Optional[str] = None,
                       limit: int = Query(100, ge=1, le=1000)):
        return await queries.findings(subject, outcome, limit)

    @app.get("/world/graph", response_model=Graph)
    async def graph(subject: Optional[str] = None, limit: int = Query(300, ge=1, le=2000)):
        return await queries.graph(subject, limit)

    @app.get("/world/runs/{run_id}", response_model=RunExport)
    async def run(run_id: str):
        out = await queries.export_run(run_id)
        if out is None:
            raise HTTPException(404, f"No run {run_id} in the world model.")
        return out

    @app.get("/world/claims/{claim_id}", response_model=ClaimOut)
    async def claim(claim_id: str):
        out = await queries.claim(claim_id)
        if out is None:
            raise HTTPException(404, f"No claim {claim_id} in the world model.")
        return out

    @app.get("/world/brief")
    async def brief(q: str = Query(..., min_length=3), limit: int = Query(8, ge=1, le=50)):
        found = await queries.findings_for(q, None, limit)
        return {"q": q, "findings": [f.model_dump() for f in found], "lines": [f.line() for f in found]}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "8791")))
