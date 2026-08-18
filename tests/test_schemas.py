import pytest
from pydantic import ValidationError
from src.api.schemas import QueryRequest, RawDocument, DocumentMetadata

def test_valid_query_request():
    """Test that a properly formatted search request is accepted."""
    req = QueryRequest(
        query="How does QuantumStrike achieve low latency?",
        tenant_id="tenant-Alpha",
        user_role="quant",
        top_k=3
    )
    assert req.tenant_id == "tenant-Alpha"
    assert req.user_role == "quant"
    assert req.top_k == 3

def test_invalid_query_request_missing_role():
    """Test that the API strictly rejects a request missing security credentials."""
    with pytest.raises(ValidationError):
        # Missing 'user_role' which is required for RBAC!
        QueryRequest(
            query="Tell me about the architecture",
            tenant_id="tenant-Alpha"
        )

def test_valid_document_metadata():
    """Test that document ingestion strictly requires tenant isolation tags."""
    meta = DocumentMetadata(
        tenant_id="tenant-Beta",
        allowed_roles=["admin"],
        source="wiki"
    )
    assert meta.tenant_id == "tenant-Beta"
    assert "admin" in meta.allowed_roles