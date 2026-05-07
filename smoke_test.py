from __future__ import annotations

from fastapi.testclient import TestClient

from dspy_openai_endpoint.main import app


client = TestClient(app)


response = client.get("/healthz")
print("healthz", response.status_code, response.json())

response = client.get("/v1/models")
print("models", response.status_code, len(response.json()["data"]))
