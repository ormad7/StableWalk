"""Tests for session registry and autosave preferences."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stablewalk.io import session_registry as registry


class SessionRegistryTests(unittest.TestCase):
    def test_remember_and_list_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = root / "sessions"
            sessions.mkdir()
            bundle = sessions / "stablewalk_session_demo"
            bundle.mkdir()
            (bundle / "session_metadata.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(registry.config, "SESSION_EXPORT_DIR", sessions), mock.patch.object(
                registry.config, "ensure_output_dirs", lambda: None
            ):
                registry.remember_session(
                    bundle,
                    display_name="Demo Walk",
                    video_source="walk.mp4",
                    frame_count=40,
                )
                recent = registry.list_recent_sessions()
                self.assertEqual(len(recent), 1)
                self.assertEqual(recent[0].display_name, "Demo Walk")
                self.assertTrue(registry.is_autosave_enabled())
                registry.set_autosave_enabled(False)
                self.assertFalse(registry.is_autosave_enabled())
                registry.set_autosave_enabled(True)


if __name__ == "__main__":
    unittest.main()
