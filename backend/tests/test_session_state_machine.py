from agent_core.supervisor.session_state_machine import SessionPhase, SessionStateMachine


def test_starts_listening():
    sm = SessionStateMachine()
    assert sm.phase == SessionPhase.LISTENING


def test_barge_in_from_thinking_returns_true_and_resets():
    sm = SessionStateMachine()
    sm.transition_to(SessionPhase.THINKING)
    assert sm.barge_in() is True
    assert sm.phase == SessionPhase.LISTENING


def test_barge_in_from_speaking_returns_true_and_resets():
    sm = SessionStateMachine()
    sm.transition_to(SessionPhase.SPEAKING)
    assert sm.barge_in() is True
    assert sm.phase == SessionPhase.LISTENING


def test_barge_in_while_already_listening_is_a_no_op_returning_false():
    sm = SessionStateMachine()
    assert sm.barge_in() is False
    assert sm.phase == SessionPhase.LISTENING


def test_double_barge_in_from_any_state_never_gets_stuck():
    sm = SessionStateMachine()
    sm.transition_to(SessionPhase.SPEAKING)
    assert sm.barge_in() is True
    assert sm.barge_in() is False  # already listening — idempotent, no crash
    sm.transition_to(SessionPhase.THINKING)
    assert sm.barge_in() is True
    assert sm.phase == SessionPhase.LISTENING
