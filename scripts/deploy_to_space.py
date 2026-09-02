"""Upload the working tree (code + baked-in artifacts) to a Hugging Face Space.

Used by .github/workflows/deploy-hf-space.yml. Uses huggingface_hub rather
than `git push` so it is immune to shallow-clone rejection and handles Git
LFS for the .parquet / .joblib artifacts automatically.

Env:
    HF_TOKEN     - Hugging Face token with write access (required)
    HF_USERNAME  - Space owner (required)
    HF_SPACE     - Space name (required)
    DEPLOY_MSG   - commit message (optional)
"""

from __future__ import annotations

import os
import shutil
import sys

from huggingface_hub import HfApi

IGNORE = [
    ".git*", ".github/**", "deploy/**", "docs/**", "tests/**",
    ".venv/**", "venv/**", "logs/**", "scratchpad/**", "Makefile",
    "__pycache__/**", "**/__pycache__/**", "*.pyc",
    ".env", ".ruff_cache/**", ".pytest_cache/**", ".idea/**", ".vscode/**",
    "data/raw/**", "data/processed/features.parquet",
    "data/processed/last_update_report.json",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    user = os.environ.get("HF_USERNAME")
    space = os.environ.get("HF_SPACE")
    if not (token and user and space):
        sys.exit("HF_TOKEN / HF_USERNAME / HF_SPACE must all be set")

    repo_id = f"{user}/{space}"

    # The Space reads its config (sdk: docker, app_port) from README front-matter.
    shutil.copyfile("deploy/SPACE_README.md", "README.md")

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=".",
        commit_message=os.environ.get("DEPLOY_MSG", "deploy from CI"),
        ignore_patterns=IGNORE,
        delete_patterns=["*", "**/*"],   # keep the Space in sync with this tree
    )
    print(f"Deployed to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
