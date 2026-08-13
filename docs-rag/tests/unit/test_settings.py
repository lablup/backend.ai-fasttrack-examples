"""The layered .env loader decides where secrets come from — pin the order."""

from __future__ import annotations

import os

import pytest

from docs_rag.settings import (
    Settings,
    clear_unset_env,
    is_unset,
    load_layered_dotenv,
    parse_dotenv,
)


def test_parse_keeps_hash_inside_a_value():
    """An inline '#' must not be treated as a comment.

    A '#' is legal inside a password or a URL fragment, and silently truncating
    one corrupts a secret in a way that is very hard to trace.
    """
    parsed = parse_dotenv("PASSWORD=abc#def\n# a real comment\nEMPTY=\n")
    assert parsed["PASSWORD"] == "abc#def"
    assert parsed["EMPTY"] == ""
    assert "# a real comment" not in parsed


def test_parse_strips_surrounding_quotes_and_skips_junk():
    parsed = parse_dotenv('A="quoted"\nB=\'single\'\nnot a pair\n\n')
    assert parsed == {"A": "quoted", "B": "single"}


def test_earlier_layer_wins_and_environment_beats_every_file(tmp_path, monkeypatch):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    first.write_text("SHARED=from_first\nONLY_FIRST=1\n")
    second.write_text("SHARED=from_second\nONLY_SECOND=2\n")

    monkeypatch.delenv("SHARED", raising=False)
    monkeypatch.delenv("ONLY_FIRST", raising=False)
    monkeypatch.delenv("ONLY_SECOND", raising=False)
    monkeypatch.setenv("PRESET", "from_environment")

    loaded = load_layered_dotenv([first, second])

    assert loaded == [first, second]
    assert os.environ["SHARED"] == "from_first"
    # A later layer still supplies keys no earlier layer had.
    assert os.environ["ONLY_SECOND"] == "2"
    assert os.environ["PRESET"] == "from_environment"


def test_missing_files_are_skipped_not_fatal(tmp_path):
    """/models does not exist on a laptop, and must not break a local run."""
    assert load_layered_dotenv([tmp_path / "nope.env"]) == []


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "${{ secrets.OPENAI_API_KEY }}", "${{secrets.MISSING}}"],
)
def test_blank_and_unresolved_placeholders_count_as_unset(value):
    assert is_unset(value)


@pytest.mark.parametrize("value", ["sk-real", "0", "false", "a ${{ b }} c"])
def test_real_values_count_as_set(value):
    """Only a value that is *entirely* a placeholder is discounted.

    '0' and 'false' are meaningful settings values and must survive.
    """
    assert not is_unset(value)


def test_a_blank_in_the_pipeline_yaml_falls_through_to_the_env_file(tmp_path, monkeypatch):
    """The YAML ships every setting blank for the user to fill in.

    A cluster that propagates `envs` exports those blanks. If a blank counted as
    "set", it would beat the .env in model storage and no task would ever see a
    key.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_the_env_file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    load_layered_dotenv([env_file])

    assert os.environ["OPENAI_API_KEY"] == "from_the_env_file"


def test_an_unresolved_secret_reference_falls_through_too(tmp_path, monkeypatch):
    """A reference to a secret that does not exist arrives as literal text."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_the_env_file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "${{ secrets.NOT_DEFINED }}")

    load_layered_dotenv([env_file])

    assert os.environ["OPENAI_API_KEY"] == "from_the_env_file"


def test_a_value_typed_into_the_pipeline_yaml_wins_over_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from_the_env_file\n")
    monkeypatch.setenv("LLM_MODEL", "typed_into_the_yaml")

    load_layered_dotenv([env_file])

    assert os.environ["LLM_MODEL"] == "typed_into_the_yaml"


def test_blank_numeric_settings_do_not_reach_an_int_field(monkeypatch):
    """`EVAL_SAMPLE_SIZE: ""` in the YAML must not crash Settings on int('')."""
    monkeypatch.setenv("EVAL_SAMPLE_SIZE", "")
    monkeypatch.setenv("GLOBAL_TOP_K", "   ")

    clear_unset_env(["EVAL_SAMPLE_SIZE", "GLOBAL_TOP_K"])
    settings = Settings()

    assert settings.eval_sample_size == 0
    assert settings.global_top_k == 15


def test_settings_read_upper_case_environment_names(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    monkeypatch.setenv("GLOBAL_TOP_K", "42")
    settings = Settings()
    assert settings.llm_model == "some-other-model"
    assert settings.global_top_k == 42


def test_thinking_body_only_sent_to_a_custom_endpoint():
    """OpenAI rejects unknown body keys, so the flag must be gated."""
    default = Settings(openai_base_url="")
    custom = Settings(openai_base_url="https://example.invalid/v1", enable_thinking=True)

    assert "extra_body" not in default.llm_kwargs()
    assert custom.llm_kwargs()["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
