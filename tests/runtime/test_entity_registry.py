import json

from app.runtime.entity_registry import EntityRegistry


def build_registry(tmp_path):
    config = {
        "entity_types": {
            "platform": {
                "fuzzy_enabled": True,
                "fuzzy_threshold": 0.70,
                "aliases": {
                    "ctrader": "cTrader",
                    "c trader": "cTrader",
                    "mt5": "MT5",
                    "metatrader": "MT5",
                },
            },
            "country": {
                "fuzzy_enabled": False,
                "fuzzy_threshold": 0.90,
                "aliases": {
                    "afghanistan": "AF",
                    "afghan": "AF",
                    "افغانستان": "AF",
                    "iran": "IR",
                    "ایران": "IR",
                },
            },
        },
        "context_field_entity_types": {
            "kyc_document_country": "country",
            "country_of_residence": "country",
        },
        "conversation": {
            "short_followup_max_tokens": 4,
            "correction_markers": [
                "نه منظورم",
                "i mean",
            ],
        },
    }

    path = tmp_path / "entities.json"

    path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return EntityRegistry.from_file(path)


def test_exact_platform_alias(tmp_path):
    registry = build_registry(tmp_path)

    result = registry.resolve_entity(
        entity_type="platform",
        text="How do I use MT5?",
    )

    assert result == "MT5"


def test_ctrader_alias(tmp_path):
    registry = build_registry(tmp_path)

    result = registry.resolve_entity(
        entity_type="platform",
        text="I want cTrader",
    )

    assert result == "cTrader"


def test_platform_typo_can_be_resolved(tmp_path):
    registry = build_registry(tmp_path)

    result = registry.resolve_entity(
        entity_type="platform",
        text="ctarder",
    )

    assert result == "cTrader"


def test_country_alias(tmp_path):
    registry = build_registry(tmp_path)

    result = registry.resolve_entity(
        entity_type="country",
        text="افغانستان",
    )

    assert result == "AF"


def test_country_fuzzy_matching_is_disabled(tmp_path):
    registry = build_registry(tmp_path)

    result = registry.resolve_entity(
        entity_type="country",
        text="afghanstan",
    )

    assert result is None


def test_context_field_maps_to_entity_type(tmp_path):
    registry = build_registry(tmp_path)

    result = (
        registry.entity_type_for_context_field(
            "kyc_document_country"
        )
    )

    assert result == "country"


def test_unknown_context_field_returns_none(tmp_path):
    registry = build_registry(tmp_path)

    result = (
        registry.entity_type_for_context_field(
            "unknown_field"
        )
    )

    assert result is None


def test_conversation_settings_are_loaded(tmp_path):
    registry = build_registry(tmp_path)

    assert (
        registry.short_followup_max_tokens()
        == 4
    )

    assert "نه منظورم" in (
        registry.correction_markers()
    )


def test_normalization_handles_persian_characters():
    result = EntityRegistry.normalize_text(
        "  كیف   يک  "
    )

    assert result == "کیف یک"