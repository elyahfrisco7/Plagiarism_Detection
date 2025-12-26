# tests/test_routes.py
from unittest.mock import patch, MagicMock

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200

@patch("app.routes.SessionLocal")
def test_theses_route(mock_session, client):
    # Mock database query
    mock_db = MagicMock()
    mock_session.return_value = mock_db
    mock_db.query.return_value.order_by.return_value.all.return_value = []
    
    response = client.get('/theses')
    assert response.status_code == 200
    mock_db.close.assert_called_once()

@patch("app.routes.SessionLocal")
def test_compare_route_missing_ids(mock_session, client):
    response = client.get('/compare')
    assert response.status_code == 400

@patch("app.routes.SessionLocal")
def test_compare_route_not_found(mock_session, client):
    mock_db = MagicMock()
    mock_session.return_value = mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    response = client.get('/compare?id1=1&id2=2')
    assert response.status_code == 404
