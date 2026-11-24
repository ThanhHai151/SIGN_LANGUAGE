from fastapi.testclient import TestClient


def test_production_healthz_is_fast_and_never_prompts_basic_auth(monkeypatch):
    # Even if a stale VSL_PREVIEW_PASSWORD env var remains configured on Render,
    # main must never return 401 Basic Auth.
    monkeypatch.setenv("VSL_PREVIEW_PASSWORD", "stale-secret")

    import backend.production as prod
    from backend.vision import qipedc_live

    def fail_if_called():
        raise AssertionError("healthz must not synchronously load live_index.npz")

    monkeypatch.setattr(qipedc_live, "indexed_entry_ids", fail_if_called)

    with TestClient(prod.app) as client:
        res_get = client.get("/healthz")
        assert res_get.status_code == 200
        body = res_get.json()
        assert body["ready"] is True
        assert body["qipedc_index_available"] is True
        assert "www-authenticate" not in {k.lower() for k in res_get.headers}

        res_head = client.head("/healthz")
        assert res_head.status_code == 200

        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "www-authenticate" not in {k.lower() for k in res_root.headers}
