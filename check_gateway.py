import requests, json
import os
URL = os.environ["GATEWAY_URL"]  # export GATEWAY_URL first (see .env.example)
h = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}
r = requests.post(URL, headers=h, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
print(r.status_code)
try:
    print([t["name"] for t in r.json()["result"]["tools"]])
except Exception:
    print(r.text[:2000])
