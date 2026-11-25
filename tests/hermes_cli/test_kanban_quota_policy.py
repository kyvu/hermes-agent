"""Quota windows must not become six-minute review respawn storms."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS', '300')
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.setattr(kb, '_pid_alive', lambda _pid: False)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def fail_review(conn, monkeypatch, now, text):
    monkeypatch.setattr(kbd.time, 'time', lambda: now)
    tid = kb.create_task(conn, title='quota review', assignee='builder')
    claim = kb.claim_task(conn, tid)
    kb.request_review(conn, tid, summary='exact head held', metadata={'head_sha': 'abc123'},
                      reviewer='reviewer', expected_run_id=claim.current_run_id)
    review = kb.claim_review_task(conn, tid)
    pid = 77123
    conn.execute('UPDATE tasks SET worker_pid=? WHERE id=?', (pid, tid))
    kb._append_event(conn, tid, 'worker_result', {'exit_code': 75, 'last_output': text},
                     run_id=review.current_run_id)
    conn.commit()
    monkeypatch.setattr(kbd, '_classify_worker_exit', lambda _pid: ('rate_limited', 75))
    kbd.detect_crashed_workers(conn)
    return tid


def test_weekly_reset_survives_reap_and_repeated_ticks(board, monkeypatch):
    now = datetime(2026, 11, 24, 20, tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
    reset = datetime(2026, 11, 28, 3, tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
    tid = fail_review(board, monkeypatch, now,
                      "You've hit your weekly limit · resets Nov 28 at 3am (America/Los_Angeles)")
    for clock in (now + 400, now + 3600, reset - 1):
        monkeypatch.setattr(kbd.time, 'time', lambda: clock)
        assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'
        task = kb.get_task(board, tid)
        assert task.status == 'review' and task.consecutive_failures == 0
    monkeypatch.setattr(kbd.time, 'time', lambda: reset)
    assert kbd.check_respawn_guard(board, tid, lane='review') is None
    handoff = board.execute("SELECT metadata FROM task_runs WHERE task_id=? AND outcome='review_requested'", (tid,)).fetchone()
    assert json.loads(handoff['metadata'])['head_sha'] == 'abc123'


def test_explicit_retry_after_not_capped_to_short_cooldown(board, monkeypatch):
    tid = fail_review(board, monkeypatch, 5_000_000, 'Rate limited. Retry after 7200 seconds')
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_007_199)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_007_200)
    assert kbd.check_respawn_guard(board, tid, lane='review') is None


def test_old_provider_failure_does_not_hold_new_authorized_override(board, monkeypatch):
    tid = fail_review(board, monkeypatch, 5_000_000, 'weekly quota exhausted')
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_001)
    kb.set_model_override(board, tid, 'already-authorized', provider='other-provider')
    assert kbd.check_respawn_guard(board, tid, lane='review') is None
    assert kb.get_task(board, tid).status == 'review'


def test_legacy_iso_ended_at_does_not_crash_dispatch(board, monkeypatch):
    tid = fail_review(board, monkeypatch, 5_000_000, 'HTTP 429')
    board.execute("UPDATE task_runs SET ended_at='1970-02-27T20:53:20+00:00', metadata=NULL "
                  "WHERE task_id=? AND outcome='rate_limited'", (tid,))
    board.commit()
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_100)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'


def test_successful_handoff_supersedes_old_provider_auth_stamp(board, monkeypatch):
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_000)
    tid = kb.create_task(board, title='handoff', assignee='builder')
    claim = kb.claim_task(board, tid)
    board.execute('UPDATE tasks SET last_failure_error=? WHERE id=?', ('quota auth failure', tid))
    board.commit()
    kb.request_review(board, tid, summary='done', reviewer='reviewer', expected_run_id=claim.current_run_id)
    assert kbd.check_respawn_guard(board, tid, lane='review') is None


def test_repeated_unknown_limits_back_off_without_failure_or_route_change(board, monkeypatch):
    tid = fail_review(board, monkeypatch, 5_000_000, 'HTTP 429')
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_301)
    assert kbd.check_respawn_guard(board, tid, lane='review') is None
    review = kb.claim_review_task(board, tid)
    board.execute('UPDATE tasks SET worker_pid=? WHERE id=?', (77123, tid))
    kb._append_event(board, tid, 'worker_result', {'exit_code': 75, 'last_output': 'HTTP 429'}, run_id=review.current_run_id)
    board.commit()
    kbd.detect_crashed_workers(board)
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_602)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'
    assert kb.get_task(board, tid).consecutive_failures == 0


@pytest.mark.parametrize('hint', [
    'Retry-After: Wed, 25 Nov 2026 12:00:00 GMT',
    '{"reset_at": "2026-11-25T12:00:00+00:00"}',
    '{"reset_at": 1795608000}',
    'resets in 12 hours',
])
def test_absolute_and_relative_reset_hints(board, monkeypatch, hint):
    now = datetime(2026, 11, 25, tzinfo=ZoneInfo('UTC')).timestamp()
    tid = fail_review(board, monkeypatch, now, hint)
    monkeypatch.setattr(kbd.time, 'time', lambda: now + 43199)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'
    monkeypatch.setattr(kbd.time, 'time', lambda: now + 43200)
    assert kbd.check_respawn_guard(board, tid, lane='review') is None


def test_model_change_on_same_provider_does_not_bypass_subscription_quota(board, monkeypatch):
    tid = fail_review(board, monkeypatch, 5_000_000, 'weekly quota exhausted')
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_001)
    # An unspecified provider still uses the same profile's provider.
    kb.set_model_override(board, tid, 'another-model')
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'


@pytest.mark.parametrize('hint', ['Retry-After: 0', '{"reset_at": 4999999}'])
def test_fresh_failure_with_nonfuture_hint_uses_backoff(board, monkeypatch, hint):
    tid = fail_review(board, monkeypatch, 5_000_000, hint)
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_001)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'


@pytest.mark.parametrize('old_provider,new_provider', [(None, 'provider-a'), ('provider-a', None)])
def test_implicit_and_explicit_same_provider_cannot_escape_quota(board, monkeypatch, old_provider, new_provider):
    from hermes_cli.kanban_quota import policy
    monkeypatch.setattr(policy, '_configured_provider', lambda _profile: 'provider-a')
    tid = fail_review(board, monkeypatch, 5_000_000, 'weekly quota exhausted')
    if old_provider:
        event = board.execute("SELECT id,payload FROM task_events WHERE task_id=? AND kind='created'", (tid,)).fetchone()
        payload = json.loads(event['payload']) | {'provider_override': old_provider, 'model_override': 'm'}
        board.execute('UPDATE task_events SET payload=? WHERE id=?', (json.dumps(payload), event['id']))
    kb.set_model_override(board, tid, 'm' if new_provider else None, provider=new_provider)
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_001)
    assert kbd.check_respawn_guard(board, tid, lane='review') == 'rate_limit_cooldown'


def test_real_worker_result_preserves_structured_reset(board, monkeypatch):
    from gateway import session_kanban
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_000_000)
    tid = kb.create_task(board, title='structured wait', assignee='reviewer')
    task = kb.claim_task(board, tid)
    context = {'task_id': tid, 'run_id': task.current_run_id, 'claim_lock': task.claim_lock,
               'db': board.execute('PRAGMA database_list').fetchone()[2]}
    kb._append_event(board, tid, 'worker_bound', {'pid': os.getpid(), 'claim_lock': task.claim_lock},
                     run_id=task.current_run_id)
    board.execute('UPDATE tasks SET worker_pid=? WHERE id=?', (77123, tid))
    board.commit()
    refusal = {'final_response': 'usage limit resets in ~12h.', 'failed': True,
               'failure_reason': 'rate_limit', 'failure_retryable': True,
               'failure_resets_at': 5_043_200, 'provider': 'provider-a'}
    monkeypatch.setattr(session_kanban, '_run_task_turns', lambda *_args: refusal)
    frame = {'policy': {'kanban_json': json.dumps(context)}}
    assert session_kanban._run_worker_turns(SimpleNamespace(), frame, []) == refusal
    result = board.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_result'", (tid,)).fetchone()
    assert json.loads(result['payload'])['failure_resets_at'] == 5_043_200
    monkeypatch.setattr(kbd, '_classify_worker_exit', lambda _pid: ('rate_limited', 75))
    kbd.detect_crashed_workers(board)
    monkeypatch.setattr(kbd.time, 'time', lambda: 5_043_199)
    assert kbd.check_respawn_guard(board, tid) == 'rate_limit_cooldown'
    session_kanban._TURNS_DONE.clear()
