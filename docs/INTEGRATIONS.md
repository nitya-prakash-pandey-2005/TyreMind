# MCP and RAG: what they are for here

Both exist to solve a problem the dashboard cannot. Neither is included because
it is fashionable, and this document says plainly what each buys and what it does
not.

---

## MCP — the model as a tool an agent can call

### The problem

A dashboard requires someone to be looking at it. During a session a race
engineer is doing five things at once, and the useful interaction is not
"navigate to a screen and read a chart" — it is asking a question and getting an
answer.

There is a second, larger problem. Every team already has analysts, and every
analyst already has an AI assistant. If TyreMind is a website, it competes for
attention. If it is a **tool their assistant can call**, it becomes part of a
workflow they already have.

### What it is

`src/tyremind/mcp_server.py` exposes the estimator over the [Model Context
Protocol](https://modelcontextprotocol.io), so Claude — or any MCP client — can
query it directly.

```bash
python -m tyremind.mcp_server            # stdio, for Claude Desktop
python -m tyremind.mcp_server --http     # streamable HTTP
```

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "tyremind": {
      "command": "python",
      "args": ["-m", "tyremind.mcp_server"],
      "cwd": "D:/TrackShift Innovation Challenge"
    }
  }
}
```

### The seven tools, and why only seven

| Tool | Answers |
|---|---|
| `list_sessions` | What can I analyse? |
| `get_degradation` | How fast is each compound going away? |
| `explain_lap` | Why was this lap slower — was it the tyre? |
| `project_tyre_life` | How much longer will this set last? |
| `recommend_strategy` | Should we box, and when? |
| `assess_trust` | Should I believe the answer in this situation? |
| `search_documentation` | How was this validated? What are the limits? |

The count is deliberate. The MCP practice literature is blunt about this: a
server that exposes forty tools "dumps 43 tools into the context window" and
degrades the agent before it has done anything. Seven covers the questions people
actually ask, and each returns a compact structured result rather than a dump.

### Two design decisions worth defending

**Every tool returns uncertainty.** An agent handed `degradation is 0.113` will
state it as fact. One handed `0.113 ± 0.023, and the model is extrapolating past
what it observed` can hedge correctly. Interval-free responses were never an
option.

**The server carries instructions that constrain the claim.** The `instructions`
field tells the client, in the model's own context, that TyreMind estimates a
latent performance state and *not* physical tread depth, and that two of its
three identifying assumptions are priors. Without that, a helpful assistant will
paraphrase "degradation" into "tyre wear" and quietly overclaim on our behalf.

### What it does not do

It does not let an agent change anything. Every tool is read-only. There is no
write path, no configuration surface, and nothing that could alter a stored
result.

---

## RAG — grounding the honest answer

### The problem

The most important question anyone asks is *"why should I believe this?"*, and
the honest answer is spread across a research audit, a model card, a limitations
register and seven experiment result files. Nobody reads all of that, so in
practice the question goes unanswered — or worse, gets answered from memory,
which is where confident wrong numbers come from.

### What it is

`src/tyremind/rag/index.py` indexes the project's own documentation and its
recorded results, and retrieves passages with their sources attached. The
dashboard exposes it as **Ask the method**; the MCP server exposes it as
`search_documentation`.

**It retrieves. It does not generate.** Every passage shown is lifted verbatim
from a file in the repository. A generated summary would read better and would
destroy the only property that makes this worth having: that a reader can open
the cited file and check.

### Hybrid retrieval, and why

Two retrievers with opposite failure modes, fused with **Reciprocal Rank Fusion**:

| | Good at | Blind to |
|---|---|---|
| **BM25** (lexical) | exact tokens — `CRPS`, `C-MAPSS`, `0.0044` | paraphrase |
| **Latent semantic** (TF-IDF → SVD) | paraphrase — "how sure is it" finding a passage on calibration | rare exact tokens |

RRF merges the two ranked lists without either needing to be calibrated against
the other, which is exactly why it is the standard choice. The dashboard shows
both ranks per result, so you can see which retriever found each passage.

Experiment results are indexed too, rendered as sentences rather than raw JSON —
a query for "how accurate" will never match `{"ssm_mae": 0.0044}`, but it will
match *"TyreMind recovers a known degradation rate with mean absolute error
0.0044 s/lap"*.

### The honest limitation

**The embeddings are TF-IDF plus SVD, not a transformer encoder.**

A sentence-transformer would retrieve better. It also needs torch, which is
roughly two gigabytes, in a product whose central promise is that the demo runs
with the network unplugged and works on a fresh clone. Latent semantic analysis
*is* a genuine dense semantic embedding — just a weaker one — and it ships inside
scikit-learn, which was already a dependency.

`EMBEDDING_BACKEND` in the module documents the swap for anyone who wants it. The
retrieval interface does not change.

### The relevance floor

RRF ranks everything, so without a floor the least-bad passage in the corpus
comes back with a confident citation attached. A hit must clear a minimum BM25 or
cosine score to be returned at all, and the panel says "nothing matched" rather
than showing something irrelevant.

That threshold is on *scores*, not ranks. A rank-based cutoff does nothing on a
small corpus, where every passage ranks near the top by construction — which is
exactly the bug the tests now guard against.

---

## How they fit together

```
                    ┌──────────────────────────┐
   race engineer ──▶│  Claude / any MCP client │
                    └────────────┬─────────────┘
                                 │  7 read-only tools
                    ┌────────────▼─────────────┐
                    │     TyreMind estimator   │
                    │  posterior + uncertainty │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  hybrid retrieval over   │
                    │  docs + recorded results │
                    └──────────────────────────┘
```

The estimator answers *what the tyre is doing*. Retrieval answers *whether to
believe it*. An assistant with both can give an answer and its caveat in the same
breath, which is the only form of this answer that is safe to act on.

---

## Trying it

```bash
# MCP tools, without a client
python -c "
import asyncio, json
from tyremind.mcp_server import build_server
s = build_server()
print(json.loads(asyncio.run(s.call_tool('get_degradation',
      {'session_id': '2024-monza-R'})).content[0].text))
"

# Retrieval, from the API
curl "http://127.0.0.1:8077/api/ask?q=what+does+it+not+measure"
```
