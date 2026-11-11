"""
A copy of the one Quorum file this app needs: the paper search (literature.py).

Why a copy: the hosted site on Vercel cannot reach ~/ai-scientist/quorum on
this laptop, and the rest of Quorum pulls in heavy libraries the app never
uses. The original lives at github.com/SHAFF622/quorum, quorum/literature.py.
If you change it there, copy it here again.

world/ is a copy of Quorum's world model (quorum/world): papers, findings and the
findings agents wrote, stored with Tortoise ORM in SQLite. Same rule: change it in
Quorum, then copy the folder here again.
"""
