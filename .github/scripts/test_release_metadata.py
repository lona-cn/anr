import unittest

from release_metadata import release_metadata


class ReleasePolicyTests(unittest.TestCase):
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def test_only_main_or_master_pushes_create_snapshots(self):
        for branch in ("main", "master"):
            with self.subTest(branch=branch):
                actual = release_metadata("push", f"refs/heads/{branch}", self.SHA, "0.1.0")
                self.assertTrue(actual["publish"])
                self.assertTrue(actual["prerelease"])
                self.assertEqual(actual["tag"], f"{branch}-0123456789ab")
                self.assertEqual(actual["label"], actual["tag"])
        for event, ref in (
            ("pull_request", "refs/heads/main"),
            ("workflow_dispatch", "refs/heads/main"),
            ("push", "refs/heads/feature"),
            ("push", "refs/tags/model-dfn3-fe5eb64fa2e4"),
        ):
            with self.subTest(event=event, ref=ref):
                self.assertFalse(release_metadata(event, ref, self.SHA, "0.1.0")["publish"])

    def test_version_tag_must_match_the_workspace(self):
        actual = release_metadata("push", "refs/tags/v0.1.0", self.SHA, "0.1.0")
        self.assertTrue(actual["publish"])
        self.assertFalse(actual["prerelease"])
        self.assertEqual((actual["tag"], actual["label"]), ("v0.1.0", "0.1.0"))
        with self.assertRaises(ValueError):
            release_metadata("push", "refs/tags/v0.2.0", self.SHA, "0.1.0")

    def test_version_prereleases_cannot_become_stable_releases(self):
        actual = release_metadata("push", "refs/tags/v0.1.0-rc.1", self.SHA, "0.1.0-rc.1")
        self.assertTrue(actual["prerelease"])
        self.assertEqual(actual["label"], "0.1.0-rc.1")

    def test_commit_and_version_cannot_inject_paths_or_outputs(self):
        for sha, version in (("bad\ncommit", "0.1.0"), (self.SHA, "../0.1.0"), (self.SHA, "0.1.0\nother=value")):
            with self.subTest(sha=sha, version=version):
                with self.assertRaises(ValueError):
                    release_metadata("push", "refs/heads/main", sha, version)


if __name__ == "__main__":
    unittest.main()
