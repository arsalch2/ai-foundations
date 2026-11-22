# AI Foundations · Week 1

A small FastAPI service that covers the four basics every AI Engineer job assumes:
plain LLM calls, streaming, structured outputs and tool calling, plus a first evaluation.

It is the foundation for portfolio project 1 (document assistant) and project 2 (order-email agent).

| Endpoint | What it shows | Day |
| --- | --- | --- |
| `POST /chat` | One LLM call, token usage | 2 |
| `POST /chat/stream` | Token streaming over Server-Sent Events | 3 |
| `POST /extract/order` | Structured output: email → validated Pydantic object, email treated as untrusted | 4 |
| `POST /ask` | Tool-calling loop with a step limit (fake stock lookup) | 5 |
| `python -m evals.run_eval` | Field-level accuracy on 10 German/English test emails | 6 |

## Setup

```bash
git clone <your repo> && cd ai-foundations
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
pytest -q          # runs offline with a fake LLM, no key needed
```

### Pick a provider (edit `.env`)

- **Ollama (free, local):** install from ollama.com, run `ollama pull llama3.2`, keep `LLM_PROVIDER=ollama`.
  Good for learning; small local models make more extraction mistakes, which makes the eval more interesting.
- **OpenAI:** `LLM_PROVIDER=openai`, set `OPENAI_API_KEY` and a small model name. A few euros of credit covers this week.
- **Azure OpenAI:** create an Azure OpenAI resource, deploy a model, set `LLM_PROVIDER=azure`,
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `CHAT_MODEL` = the *deployment* name.
  If your resource rejects the v1 endpoint, set `AZURE_OPENAI_API_VERSION` to a dated version from the Azure docs.

The code is the same for all three: `app/llm.py` is the only place that knows the difference.

## Run

```bash
uvicorn app.main:app --reload
# open http://localhost:8000/docs for the interactive API
```

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Erkläre RAG in einem Satz."}]}'

curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to ten slowly."}]}'

curl -s localhost:8000/extract/order -H 'content-type: application/json' \
  -d '{"email_text":"Bitte liefern Sie 200 Stück 6205-2RS bis 15.01.2027. Becker GmbH"}'

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Haben Sie 200 Stück 6205-2RS auf Lager?"}'
```

## Day 3 exercise: the React side of streaming

`EventSource` only supports GET, so read the POST stream with `fetch`:

```ts
const res = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ messages: [{ role: "user", content: prompt }] }),
});
const reader = res.body!.getReader();
const decoder = new TextDecoder();
let buffer = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split("\n\n");
  buffer = events.pop()!;                       // keep the incomplete tail
  for (const block of events) {
    const [eventLine, dataLine] = block.split("\n");
    if (eventLine === "event: token") setText(t => t + JSON.parse(dataLine.slice(6)).text);
  }
}
```

Build a small Next.js chat page around this (you will need CORS: add FastAPI's `CORSMiddleware`).

## Evaluation results

Run `python -m evals.run_eval` and paste `evals/results.md` here. Keep a short log:
what you changed (prompt, model, schema) → what the scores did.

| Date | Change | is_order | company | items | date | language |
| --- | --- | --- | --- | --- | --- | --- |
| | baseline | | | | | |

## Known limitations

- The stock tool uses fake in-memory data.
- No authentication or rate limiting yet.
- Relative dates ("early next month") are deliberately returned as null.
