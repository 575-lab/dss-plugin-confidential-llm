#!/usr/bin/env python3
"""A tiny fake OpenAI-compatible server, for testing the demo without Ollama/vLLM.

  python3 spark-demo/mock_model_server.py --port 11434
"""

import argparse
import json
import time

from aiohttp import web

REPLY = "Remote attestation lets a client verify what software and hardware a server is running before trusting it."


async def chat(request):
    body = await request.json()
    if request.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
        return web.Response(status=403)  # mimic Ollama's Host check
    if not body.get("stream"):
        return web.json_response({
            "id": "mock", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for word in REPLY.split(" "):
        event = {"id": "mock", "object": "chat.completion.chunk", "model": body.get("model"),
                 "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]}
        await response.write("data: {}\n\n".format(json.dumps(event)).encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)
