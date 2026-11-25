"""Minimal client for the OpenAI Responses API with strict JSON-schema output."""
import json
import os
from urllib.request import Request, urlopen


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


STRING, NUMBER = {"type": "string"}, {"type": "number"}


def ask(instructions, payload, schema, name="kael"):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY to run the knowledge-discovery agent.")
    body = {"model": os.environ.get("OPENAI_MODEL", "gpt-5"), "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}}
    request = Request(base + "/responses", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urlopen(request, timeout=300) as response:
        answer = json.load(response)
    text = "".join(c["text"] for item in answer.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    return json.loads(text)
