from pathlib import Path

import toml


def get_current_version():
    version = "unknown"
    # adopt path to your pyproject.toml
    pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
        data = toml.load(pyproject_toml_file)
        # check project.version
        if "project" in data and "version" in data["project"]:
            version = data["project"]["version"]
        # check tool.poetry.version
        elif (
            "tool" in data
            and "poetry" in data["tool"]
            and "version" in data["tool"]["poetry"]
        ):
            version = data["tool"]["poetry"]["version"]
    return version
