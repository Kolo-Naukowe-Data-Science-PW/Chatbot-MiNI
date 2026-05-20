import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.api.prompt_builder"
# ==============================================================================
MODULE_PATH = "api.prompt_builder"


@pytest.fixture
def target_module():
    """Dynamically imports the target module."""
    import importlib

    return importlib.import_module(MODULE_PATH)


@pytest.fixture
def dummy_message_class():
    """Creates a dummy Message class to simulate the pydantic model."""

    class DummyMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

        def model_dump(self):
            return {"role": self.role, "content": self.content}

    return DummyMessage


# --- Tests for build_messages ---


def test_build_messages_basic(target_module):
    """Test building messages with only query and context (no optional args)."""
    query = "Gdzie jest dziekanat?"
    context = ["Dziekanat jest na parterze.", "Otwarty do 14:00."]

    messages = target_module.build_messages(query, context)

    assert len(messages) == 2  # System message + User message

    # 1. Check System Message
    sys_msg = messages[0]
    assert sys_msg["role"] == "system"
    assert "Jesteś pomocnym asystentem o imieniu MiNIonek" in sys_msg["content"]
    assert target_module.STATIC_FAQ in sys_msg["content"]
    # Ensure optional infos are absent
    assert "Informacja o użytkowniku:" not in sys_msg["content"]
    assert "Wskazówka dotycząca rozmówcy:" not in sys_msg["content"]

    # 2. Check User Message
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert (
        "Dziekanat jest na parterze.\n\n---\n\nOtwarty do 14:00." in user_msg["content"]
    )
    assert "Bieżące pytanie użytkownika:\nGdzie jest dziekanat?" in user_msg["content"]


def test_build_messages_with_student_info(target_module):
    """Test that field of study and semester are injected correctly."""
    messages = target_module.build_messages(
        query="Kiedy mam egzamin?",
        context=["Brak danych"],
        field_of_study="Informatyka",
        semester="3",
    )

    sys_msg = messages[0]["content"]
    assert "Użytkownik studiuje na kierunku 'Informatyka', semestr 3." in sys_msg


def test_build_messages_with_field_only(target_module):
    """Test that field of study alone (no semester) is injected correctly."""
    messages = target_module.build_messages(
        query="Kiedy mam egzamin?", context=["Brak danych"], field_of_study="Matematyka"
    )

    sys_msg = messages[0]["content"]

    # Just check for the exact sentence format that excludes the semester
    expected_phrase = (
        "Informacja o użytkowniku: Użytkownik studiuje na kierunku 'Matematyka'.\n"
    )
    assert expected_phrase in sys_msg


def test_build_messages_with_user_persona(target_module):
    """Test that valid user personas are injected correctly."""
    messages = target_module.build_messages(
        query="Jak złożyć wniosek?", context=["Brak danych"], user_type="student_junior"
    )

    sys_msg = messages[0]["content"]
    expected_hint = target_module.USER_TYPE_PERSONA["student_junior"]
    assert f"Wskazówka dotycząca rozmówcy: {expected_hint}" in sys_msg


def test_build_messages_with_invalid_persona(target_module):
    """Test that an unrecognized persona is safely ignored."""
    messages = target_module.build_messages(
        query="Test", context=["Test"], user_type="hacker"  # Not in USER_TYPE_PERSONA
    )

    sys_msg = messages[0]["content"]
    assert "Wskazówka dotycząca rozmówcy:" not in sys_msg


def test_build_messages_with_history(target_module, dummy_message_class):
    """Test that conversation history is correctly placed between system and user messages."""
    history = [
        dummy_message_class(role="user", content="Poprzednie pytanie"),
        dummy_message_class(role="assistant", content="Poprzednia odpowiedź"),
    ]

    messages = target_module.build_messages(
        query="Nowe pytanie", context=["Kontekst"], conversation_history=history
    )

    # System (1) + History (2) + Current Query (1) = 4
    assert len(messages) == 4

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Poprzednie pytanie"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Poprzednia odpowiedź"
    assert messages[3]["role"] == "user"
    assert "Nowe pytanie" in messages[3]["content"]


def test_build_messages_exception_handling(target_module):
    """Test that exceptions during build are logged and re-raised."""
    # Passing None instead of a list for 'context' will naturally trigger a
    # TypeError on "".join(context) without needing to mock built-in Python methods.
    with pytest.raises(TypeError):
        target_module.build_messages(query="Test", context=None)
