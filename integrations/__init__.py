"""Adapters that mount File_Router's cost selector onto third-party memory systems.

Each adapter does the same three things:
  1. call the host's own retriever (never replaced -- that is the control),
  2. expand each hit into the cost tiers available for it,
  3. hand the tiers to CostRouter and return what survives.

The host keeps its retrieval quality; we only change what granularity gets paid
for.  That separation is the claim being tested, so every adapter also reports
the host's un-routed baseline for the same query.
"""
