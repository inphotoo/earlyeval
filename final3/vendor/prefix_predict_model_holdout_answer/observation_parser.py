'Public-release English note.'
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObservationSignals:
    'Public-release English note.'
    # Public-release English note.
    traceback_seen: bool = False
    assertion_error: bool = False
    type_error: bool = False
    value_error: bool = False
    syntax_error: bool = False
    import_error: bool = False
    file_not_found: bool = False
    timeout: bool = False
    permission_error: bool = False
    tool_error: bool = False

    # Public-release English note.
    test_fail_seen: bool = False
    test_pass_seen: bool = False
    all_tests_passed: bool = False
    fail_count: Optional[int] = None
    pass_count: Optional[int] = None

    # Public-release English note.
    edit_failed: bool = False


# Public-release English note.
_RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE)
_RE_ASSERTION = re.compile(r"AssertionError|AssertError", re.IGNORECASE)
_RE_TYPE_ERR = re.compile(r"TypeError:")
_RE_VALUE_ERR = re.compile(r"ValueError:")
_RE_SYNTAX_ERR = re.compile(r"SyntaxError:")
_RE_IMPORT_ERR = re.compile(r"ImportError:|ModuleNotFoundError:")
_RE_FILE_NOT_FOUND = re.compile(r"FileNotFoundError:|No such file or directory")
_RE_TIMEOUT = re.compile(r"timed?\s*out|TimeoutError", re.IGNORECASE)
_RE_PERMISSION = re.compile(r"PermissionError:|Permission denied", re.IGNORECASE)
_RE_TOOL_ERROR = re.compile(
    r"command returned non-zero exit status|"
    r"Error:|"
    r"error:|"
    r"FAILED|"
    r"CalledProcessError",
)

_RE_FAIL_COUNT = re.compile(r"(\d+)\s+failed")
_RE_PASS_COUNT = re.compile(r"(\d+)\s+passed")
_RE_TEST_FAILED = re.compile(r"FAILED|FAILURES|tests?\s+failed", re.IGNORECASE)
_RE_TEST_PASSED = re.compile(r"PASSED|tests?\s+passed|OK\s*$", re.IGNORECASE)
_RE_ALL_PASSED = re.compile(
    r"all\s+tests?\s+passed|"
    r"passed\s+100%|"
    r"0\s+failed.*\d+\s+passed|"
    r"OK\s+\(\d+\s+tests?\)",
    re.IGNORECASE,
)

_RE_EDIT_FAIL = re.compile(
    r"pattern not found|"
    r"replacement was not performed|"
    r"not unique in the file|"
    r"No replacement was performed|"
    r"did not appear verbatim",
    re.IGNORECASE,
)


def parse_observation(text: Optional[str]) -> ObservationSignals:
    'Public-release English note.'
    sig = ObservationSignals()
    if not text:
        return sig

    sig.traceback_seen = bool(_RE_TRACEBACK.search(text))
    sig.assertion_error = bool(_RE_ASSERTION.search(text))
    sig.type_error = bool(_RE_TYPE_ERR.search(text))
    sig.value_error = bool(_RE_VALUE_ERR.search(text))
    sig.syntax_error = bool(_RE_SYNTAX_ERR.search(text))
    sig.import_error = bool(_RE_IMPORT_ERR.search(text))
    sig.file_not_found = bool(_RE_FILE_NOT_FOUND.search(text))
    sig.timeout = bool(_RE_TIMEOUT.search(text))
    sig.permission_error = bool(_RE_PERMISSION.search(text))
    sig.tool_error = bool(_RE_TOOL_ERROR.search(text))

    # Public-release English note.
    sig.test_fail_seen = bool(_RE_TEST_FAILED.search(text))
    sig.test_pass_seen = bool(_RE_TEST_PASSED.search(text))
    sig.all_tests_passed = bool(_RE_ALL_PASSED.search(text))

    m_fail = _RE_FAIL_COUNT.search(text)
    if m_fail:
        sig.fail_count = int(m_fail.group(1))
        sig.test_fail_seen = True

    m_pass = _RE_PASS_COUNT.search(text)
    if m_pass:
        sig.pass_count = int(m_pass.group(1))
        sig.test_pass_seen = True

    # Public-release English note.
    sig.edit_failed = bool(_RE_EDIT_FAIL.search(text))

    return sig
