"""Keep quota waits on their originating task route, without changing any route.

Fallback remains the worker's configured, bounded provider chain. The dispatcher
never invents a provider, buys credits, or changes a profile to escape a quota.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.retry_utils import reset_delay_from_message
from agent.runtime_scope import bind_profile_scope
from hermes_cli.config import load_config_readonly
from hermes_cli.profiles import resolve_profile_env
from hermes_cli.providers import _plugin_alias_owner, normalize_provider

_MAX_PROBE_WAIT = 6 * 3600
_NAMED_RESET = re.compile(
    r"resets?\s+([A-Za-z]{3})\s+(\d{1,2})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def _dict(raw):
    try:
        value = json.loads(raw or '{}')
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_route(conn, task_id, run):
    claim = conn.execute(
        "SELECT id FROM task_events WHERE task_id=? AND run_id=? AND kind='claimed' ORDER BY id LIMIT 1",
        (task_id, run['id']),
    ).fetchone()
    if claim is None:
        return None  # Old/manual fixtures without route evidence cannot prove a change.
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? AND id<=? "
        "AND kind IN ('model_override_set','created') ORDER BY id DESC LIMIT 1",
        (task_id, claim['id']),
    ).fetchone()
    data = _dict(event['payload']) if event else {}
    if event and event['kind'] == 'model_override_set':
        return (run['profile'], data.get('model'), data.get('provider'))
    return (run['profile'], data.get('model_override'), data.get('provider_override'))


def _configured_provider(profile):
    try:
        home = resolve_profile_env(profile)
    except FileNotFoundError:
        return None
    with bind_profile_scope(home, hydrate_secrets=False):
        model = (load_config_readonly() or {}).get('model', {})
    return model.get('provider') if isinstance(model, dict) else None


def route_changed(conn, task_id, run):
    """Only an explicit task reassignment/override can escape the old route's wait."""
    old = _run_route(conn, task_id, run)
    task = conn.execute('SELECT assignee, model_override, provider_override FROM tasks WHERE id=?', (task_id,)).fetchone()
    if old is None or task is None or (old[0], old[2]) == (task['assignee'], task['provider_override']):
        return False
    previous = _dict(run['metadata']).get('quota_provider') or old[2] or _configured_provider(old[0])
    current = task['provider_override'] or _configured_provider(task['assignee'])
    previous = _plugin_alias_owner(normalize_provider(previous or ''))
    current = _plugin_alias_owner(normalize_provider(current or ''))
    # Auto/absent identity is not evidence of a different authorized provider.
    # Keep its wait rather than resolving credentials or guessing auto's route.
    if previous in ('', 'auto', 'unknown') or current in ('', 'auto', 'unknown'):
        return False
    return previous != current


def _message_deadline(text, now):
    deadlines = []
    delay = reset_delay_from_message(text)
    if delay is not None and math.isfinite(delay) and delay >= 0:
        deadlines.append(now + delay)
    # Header-style diagnostics and structured body fields survive stringification.
    for match in re.finditer(r'(?:retry[-_]after(?:_seconds)?)[\s\"\x27:=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE):
        deadlines.append(now + float(match[1]))
    for match in re.finditer(r'(?:reset_at|resets_at)[\s\"\x27:=]+(\d{10,13}(?:\.\d+)?)', text, re.IGNORECASE):
        stamp = float(match[1])
        deadlines.append(stamp / 1000 if stamp > 1e12 else stamp)
    for match in re.finditer(r'(?:reset_at|resets_at)[\s\"\x27:=]+(\d{4}-\d{2}-\d{2}T[^\s\"\x27}]+)', text, re.IGNORECASE):
        try:
            stamp = datetime.fromisoformat(match[1])
            if stamp.tzinfo is not None:
                deadlines.append(stamp.timestamp())
        except ValueError:
            pass
    for match in re.finditer(r'retry-after[\s\"\x27:=]+([A-Za-z]{3},[^\r\n\"\x27}]+GMT)', text, re.IGNORECASE):
        try:
            deadlines.append(parsedate_to_datetime(match[1]).timestamp())
        except (ValueError, TypeError, OverflowError):
            pass
    match = _NAMED_RESET.search(text)
    if match:
        month, day, hour, minute, half, zone = match.groups()
        try:
            tz = ZoneInfo(zone)
            current = datetime.fromtimestamp(now, tz)
            month_num = datetime.strptime(month.title(), '%b').month
            reset = datetime(current.year, month_num, int(day), int(hour) % 12 + (12 if half.lower() == 'pm' else 0),
                             int(minute or 0), tzinfo=tz)
            # The provider omits the year. Roll only across the December/January boundary.
            if reset.timestamp() < now and current.month == 12 and month_num == 1:
                reset = reset.replace(year=current.year + 1)
            deadlines.append(reset.timestamp())
        except (ValueError, ZoneInfoNotFoundError):
            pass  # Malformed vendor hint uses the conservative bounded probe below.
    return max(deadlines, default=None)


def quota_retry_metadata(conn, task_id, run_id, now, cooldown):
    """Persist only a deadline. Never copy provider text or credentials into metadata."""
    result = conn.execute(
        "SELECT payload, created_at FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_result' ORDER BY id DESC LIMIT 1",
        (task_id, run_id),
    ).fetchone()
    payload = _dict(result['payload']) if result else {}
    text = str(payload.get('last_output', ''))
    deadline = payload.get('failure_resets_at')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        from hermes_cli.kanban_db import _to_epoch
        # Relative vendor hints start when the failure was recorded, not when reaped.
        failed_at = _to_epoch(result['created_at']) if result else None
        deadline = _message_deadline(text, failed_at if failed_at is not None else now)
    provider = payload.get('quota_provider')
    identity = {'quota_provider': provider} if isinstance(provider, str) and re.fullmatch(r'[\w-]{1,64}', provider) else {}
    if deadline is not None and deadline > now:
        return {'retry_not_before': deadline, 'quota_hint': 'provider', **identity}
    # Unknown quota: exponential probes, capped at six hours, neutral to the task
    # failure counter. A named exhausted allowance starts at the conservative cap.
    streak = 0
    for run in conn.execute('SELECT * FROM task_runs WHERE task_id=? AND id<? ORDER BY id DESC', (task_id, run_id)):
        if run['outcome'] != 'rate_limited' or route_changed(conn, task_id, run):
            break
        streak += 1
        if streak >= 16:
            break
    # Explicit zero is the existing next-tick E2E mode. Production defaults to 300s.
    wait = min(_MAX_PROBE_WAIT, max(0, cooldown) * 2 ** streak)
    if re.search(r'weekly|billing|credits|subscription|quota.{0,30}exhaust', text, re.IGNORECASE):
        wait = max(wait, _MAX_PROBE_WAIT)
    return {'retry_not_before': now + wait, 'quota_hint': 'bounded_probe', **identity}


def retry_deadline(conn, task_id, run, cooldown, ended_at):
    """Read durable hints, or reconstruct exact-run evidence for pre-upgrade runs."""
    data = _dict(run['metadata'])
    deadline = data.get('retry_not_before')
    if isinstance(deadline, (int, float)) and math.isfinite(deadline):
        return deadline
    if ended_at is None:
        return math.inf  # An unreadable legacy stamp must not create a busy probe loop.
    return quota_retry_metadata(conn, task_id, run['id'], ended_at, cooldown)['retry_not_before']
