# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
import tempfile
import os

import changelog


def test_git():
    git_version = changelog.git("--version")
    assert len(git_version) == 1
    assert re.match(r"git version", git_version[0]) is not None


def test_find_last_release():
    last_release = changelog.find_last_release_sha()
    assert re.match(r"[0-9a-f]{40}", last_release)


def test_find_commits_since():
    last_release = changelog.find_last_release_sha()
    commits = changelog.find_commits_since(last_release)
    assert isinstance(commits, list)
    assert len(commits) > 0

    for commit in commits:
        assert isinstance(commit, str)
        assert re.match(r"[0-9a-f]{40}", commit)

    assert last_release in commits[-1]


def test_parse_commits():
    commits = [
        "0" * 40 + " This is not a conventional commit",
        "1" * 40 + " fix: A conventional commit with no component",
        "2" * 40 + " fix(r/sub_dir/sub-dir): A conventional commit with a component",
    ]

    parsed = changelog.parse_commits(commits)

    # Non-conventional commits not included (same as cz ch)
    assert len(parsed) == 2

    assert parsed[0]["sha"] == "1" * 40
    assert parsed[0]["type"] == "fix"
    assert parsed[0]["component"] is None
    assert parsed[0]["message"] == "A conventional commit with no component"

    assert parsed[1]["sha"] == "2" * 40
    assert parsed[1]["type"] == "fix"
    assert parsed[1]["component"] == "r/sub_dir/sub-dir"
    assert parsed[1]["message"] == "A conventional commit with a component"


def test_group_commits_by_type():
    parsed = [
        {"type": "fix", "sha": "0"},
        {"type": "fix", "sha": "1"},
        {"type": "chore", "sha": "2"},
    ]

    grouped = changelog.group_commits_by_type(parsed)
    assert list(grouped.keys()) == ["fix", "chore"]

    assert len(grouped["fix"]) == 2
    assert grouped["fix"][0] is parsed[0]
    assert grouped["fix"][1] is parsed[1]

    assert len(grouped["chore"]) == 1
    assert grouped["chore"][0] is parsed[2]


def test_group_commits_by_top_level_component():
    parsed = [
        {"component": None, "sha": "0"},
        {"component": None, "sha": "1"},
        {"component": "r/abcd", "sha": "2"},
        {"component": "r", "sha": "3"},
    ]

    grouped = changelog.group_commits_by_top_level_component(parsed)

    assert list(grouped.keys()) == ["", "r"]
    assert len(grouped[""]) == 2
    assert grouped[""][0] is parsed[0]
    assert grouped[""][1] is parsed[1]

    assert len(grouped["r"]) == 2
    assert grouped["r"][0] is parsed[2]
    assert grouped["r"][1] is parsed[3]


def test_render():
    parsed = [
        {"type": "fix", "component": None, "message": "message 0"},
        {"type": "chore", "component": None, "message": "message 1"},
        {"type": "fix", "component": "r/abcd", "message": "message 2"},
        {"type": "fix", "component": "r", "message": "message 3"},
        {"type": "feat", "component": "r", "message": "message 4"},
    ]

    rendered = changelog.render_version_content(parsed)
    assert rendered.splitlines() == [
        "### Feat",
        "",
        "- **r**: message 4",
        "",
        "### Fix",
        "",
        "- message 0",
        "- **r/abcd**: message 2",
        "- **r**: message 3",
    ]


def test_parse_changelog():
    changelog_lines = [
        "<!-- header stuff we want untouched -->",
        "",
        "# nanoarrow Changelog",
        "",
        "## nanoarrow <some version information we want untouched>",
        "",
        "content we want untouched for each previous version",
        "",
        "## nanoarrow <some other version information we want untouched>",
        "",
        "other content we want untouched for each previous version",
    ]

    content = "\n".join(changelog_lines)
    header, version_content = changelog.parse_changelog(content)
    assert header == "<!-- header stuff we want untouched -->\n\n# nanoarrow Changelog"

    assert isinstance(version_content, dict)
    assert list(version_content.keys()) == [
        "<some version information we want untouched>",
        "<some other version information we want untouched>",
    ]

    assert list(version_content.values()) == [
        "content we want untouched for each previous version",
        "other content we want untouched for each previous version",
    ]


def test_render_new_changelog():
    changelog_lines = [
        "<!-- header stuff we want untouched -->",
        "",
        "# nanoarrow Changelog",
        "",
        "## nanoarrow <some version information we want untouched>",
        "",
        "content we want untouched for each previous version",
        "",
        "## nanoarrow <some other version information we want untouched>",
        "",
        "other content we want untouched for each previous version",
    ]

    with tempfile.TemporaryDirectory() as tempdir:
        changes_no_version = changelog.render_new_changelog()
        assert re.match(r"^## nanoarrow", changes_no_version) is None

        changes_with_version = changelog.render_new_changelog("some version info")
        assert re.match(r"^## nanoarrow some version info", changes_with_version)

        changelog_file_name = os.path.join(tempdir, "CHANGELOG.md")
        with open(changelog_file_name, "w") as f:
            f.writelines(
                [
                    "<!-- header stuff we want untouched -->\n",
                    "\n",
                    "# nanoarrow Changelog\n",
                    "\n",
                ]
            )
            f.write(changes_with_version)

        # Make sure we do not write two version items for the same version
        modified_changelog = changelog.render_new_changelog(
            "some version info", changelog_file_name
        )
        assert len(re.findall(r"\n## nanoarrow", modified_changelog)) == 1

        # Make sure do write two version items for different versions
        modified_changelog = changelog.render_new_changelog(
            "other version info", changelog_file_name
        )
        assert len(re.findall(r"\n## nanoarrow", modified_changelog)) == 2
