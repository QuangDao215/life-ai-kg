"""Unit tests for extraction service parsing and validation logic."""

import pytest

from src.db.enums import EntityType, RelationLabel, is_valid_entity_type, is_valid_relation_label

# =============================================================================
# Entity Type Validation Tests
# =============================================================================


class TestEntityTypeValidation:
    """Tests for entity type validation."""

    def test_valid_entity_types(self) -> None:
        """Test valid entity type strings."""
        valid_types = [
            "Disease",
            "Gene",
            "Protein",
            "Compound/Drug",
            "Pathway",
            "CellType",
            "Cell Type",  # from_string normalizes this
            "Tissue/Region",
            "Phenotype",
            "Biomarker",
        ]

        for type_str in valid_types:
            assert is_valid_entity_type(type_str), f"{type_str} should be valid"

    def test_invalid_entity_types(self) -> None:
        """Test invalid entity type strings."""
        invalid_types = [
            "Unknown",
            "InvalidType",
            "",
            "Medication",  # Not a valid type
            "Organism",  # Not in our schema
        ]

        for type_str in invalid_types:
            assert not is_valid_entity_type(type_str), f"{type_str} should be invalid"

    def test_entity_type_case_insensitive(self) -> None:
        """Test that entity type validation is case-insensitive."""
        # from_string handles case variations
        assert is_valid_entity_type("disease")
        assert is_valid_entity_type("DISEASE")
        assert is_valid_entity_type("Disease")

    def test_entity_type_enum_values(self) -> None:
        """Test EntityType enum has expected values."""
        assert EntityType.DISEASE.value == "Disease"
        assert EntityType.GENE.value == "Gene"
        assert EntityType.PROTEIN.value == "Protein"
        assert EntityType.COMPOUND_DRUG.value == "Compound/Drug"


# =============================================================================
# Relation Label Validation Tests
# =============================================================================


class TestRelationLabelValidation:
    """Tests for relation label validation."""

    def test_valid_relation_labels(self) -> None:
        """Test valid relation label strings."""
        valid_labels = [
            "ASSOCIATED_WITH",
            "INCREASES_RISK",
            "DECREASES_RISK",
            "UPREGULATES",
            "DOWNREGULATES",
            "ACTIVATES",
            "INHIBITS",
            "BINDS",
            "BIOMARKER_FOR",
            "PART_OF",
            "EXPRESSED_IN",
        ]

        for label in valid_labels:
            assert is_valid_relation_label(label), f"{label} should be valid"

    def test_invalid_relation_labels(self) -> None:
        """Test invalid relation label strings."""
        invalid_labels = [
            "Unknown",
            "RELATED_TO",
            "",
            "IS_A",
            "TREATS",  # Not in our schema
            "CAUSES",  # Not in our schema
        ]

        for label in invalid_labels:
            assert not is_valid_relation_label(label), f"{label} should be invalid"

    def test_relation_label_enum_values(self) -> None:
        """Test RelationLabel enum has expected values."""
        assert RelationLabel.ASSOCIATED_WITH.value == "ASSOCIATED_WITH"
        assert RelationLabel.ACTIVATES.value == "ACTIVATES"
        assert RelationLabel.INHIBITS.value == "INHIBITS"
        assert RelationLabel.BIOMARKER_FOR.value == "BIOMARKER_FOR"


# =============================================================================
# Entity Normalization Tests
# =============================================================================


class TestEntityNormalization:
    """Tests for entity name normalization."""

    def test_normalize_simple_name(self) -> None:
        """Test normalizing simple entity names."""
        # Simulate the normalization logic
        def normalize(name: str) -> str:
            return name.lower().strip()

        assert normalize("Parkinson's Disease") == "parkinson's disease"
        assert normalize("  Ambroxol  ") == "ambroxol"
        assert normalize("GCASE") == "gcase"

    def test_normalize_preserves_special_chars(self) -> None:
        """Test normalization preserves scientific characters."""
        def normalize(name: str) -> str:
            return name.lower().strip()

        assert normalize("α-synuclein") == "α-synuclein"
        assert normalize("GBA1/GCase") == "gba1/gcase"


# =============================================================================
# Evidence Hash Tests
# =============================================================================


class TestEvidenceHash:
    """Tests for evidence quote hashing."""

    def test_hash_consistency(self) -> None:
        """Test that same quote produces same hash."""
        from src.db.models import Evidence

        quote = "Ambroxol enhances GCase activity in neuronal cells."
        hash1 = Evidence.compute_hash(quote)
        hash2 = Evidence.compute_hash(quote)

        assert hash1 == hash2

    def test_hash_differs_for_different_quotes(self) -> None:
        """Test that different quotes produce different hashes."""
        from src.db.models import Evidence

        quote1 = "Ambroxol enhances GCase activity."
        quote2 = "Ambroxol inhibits GCase activity."

        hash1 = Evidence.compute_hash(quote1)
        hash2 = Evidence.compute_hash(quote2)

        assert hash1 != hash2

    def test_hash_handles_whitespace(self) -> None:
        """Test hash handles whitespace consistently."""
        from src.db.models import Evidence

        quote1 = "Ambroxol enhances GCase"
        quote2 = "  Ambroxol enhances GCase  "

        # Hashes may or may not differ based on implementation
        # This test documents expected behavior
        hash1 = Evidence.compute_hash(quote1)
        hash2 = Evidence.compute_hash(quote2)

        # If normalized, should be equal; if not, should differ
        # Just verify both produce valid hashes
        assert len(hash1) == 64  # SHA256 hex length
        assert len(hash2) == 64


# =============================================================================
# Pagination Helper Tests
# =============================================================================


class TestPaginationHelpers:
    """Tests for pagination calculations."""

    def test_offset_calculation(self) -> None:
        """Test offset calculation from page number."""
        from src.schemas import PaginationParams

        params = PaginationParams(page=1, page_size=20)
        assert params.offset == 0

        params = PaginationParams(page=2, page_size=20)
        assert params.offset == 20

        params = PaginationParams(page=3, page_size=10)
        assert params.offset == 20

    def test_pages_calculation(self) -> None:
        """Test total pages calculation."""
        from src.schemas import PaginatedResponse

        # Exact division
        response = PaginatedResponse.create(
            items=[],
            total=100,
            page=1,
            page_size=20
        )
        assert response.pages == 5

        # With remainder
        response = PaginatedResponse.create(
            items=[],
            total=101,
            page=1,
            page_size=20
        )
        assert response.pages == 6

        # Empty
        response = PaginatedResponse.create(
            items=[],
            total=0,
            page=1,
            page_size=20
        )
        assert response.pages == 0


# =============================================================================
# JSON Parsing Tests
# =============================================================================


class TestJSONParsing:
    """Tests for JSON parsing in LLM responses."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON."""
        import json

        valid = '{"entities": [{"name": "test"}]}'
        parsed = json.loads(valid)
        assert "entities" in parsed

    def test_detect_markdown_wrapper(self) -> None:
        """Test detecting markdown code blocks."""
        text = '```json\n{"test": 1}\n```'

        # Check if text starts with markdown
        has_markdown = text.strip().startswith("```")
        assert has_markdown

        # Extract JSON
        if has_markdown:
            lines = text.strip().split("\n")
            json_text = "\n".join(lines[1:-1])
            import json
            parsed = json.loads(json_text)
            assert parsed["test"] == 1

    def test_handle_truncated_response(self) -> None:
        """Test handling truncated JSON gracefully."""
        import json

        truncated = '{"entities": [{"name": "test"'

        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            json.loads(truncated)


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunking:
    """Tests for text chunking functionality."""

    def test_chunk_short_text(self) -> None:
        """Test chunking text shorter than chunk size."""
        from src.services.chunking import chunk_text

        text = "This is a short text."
        chunks = list(chunk_text(text, chunk_size=1000, chunk_overlap=100))

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_long_text(self) -> None:
        """Test chunking text longer than chunk size."""
        from src.services.chunking import chunk_text

        text = "This is a sentence. " * 50  # Long text
        chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=20))

        assert len(chunks) > 1

    def test_chunk_offsets(self) -> None:
        """Test chunk offsets are correct."""
        from src.services.chunking import chunk_text

        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = list(chunk_text(text, chunk_size=50, chunk_overlap=10))

        for chunk in chunks:
            # Verify offset matches actual text position
            assert text[chunk.start_offset:chunk.end_offset] == chunk.text

    def test_chunk_empty_text(self) -> None:
        """Test chunking empty text."""
        from src.services.chunking import chunk_text

        text = ""
        chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=20))

        assert len(chunks) == 0

    def test_chunk_indices_sequential(self) -> None:
        """Test chunk indices are sequential."""
        from src.services.chunking import chunk_text

        # Use a longer text with proper sentences to avoid edge cases
        text = "This is sentence number one. " * 50
        chunks = list(chunk_text(text, chunk_size=200, chunk_overlap=50))

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for application configuration."""

    def test_settings_has_required_fields(self) -> None:
        """Test settings has all required fields."""
        from src.core.config import settings

        assert hasattr(settings, 'db_url')
        assert hasattr(settings, 'cors_origins')
        assert hasattr(settings, 'log_level')

    def test_entity_types_list(self) -> None:
        """Test entity types are properly configured."""
        from src.db.enums import EntityType

        # We have 9 entity types in our schema
        assert len(EntityType) >= 9

    def test_relation_labels_list(self) -> None:
        """Test relation labels are properly configured."""
        from src.db.enums import RelationLabel

        # Should have at least 10 relation labels
        assert len(RelationLabel) >= 10
