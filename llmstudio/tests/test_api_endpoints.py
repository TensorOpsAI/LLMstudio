import pytest


@pytest.mark.asyncio
async def test_health_check(test_engine_app):
    response = await test_engine_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Engine is up and running",
    }
