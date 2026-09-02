# Deploying to Hugging Face Spaces (Option A)

**Model:** the Space is a thin, read-only viewer. All data and model artifacts
are refreshed by a scheduled GitHub Action and force-pushed to the Space, which
rebuilds its Docker image and serves the dashboard. The Space itself holds **no
API key**.

```
CollegeFootballData ──▶ GitHub Action (every 6 h)
                         │  scripts/update_dashboard.py
                         │  → fetch, retrain, simulate, write artifacts
                         ▼
                    git push --force ──▶ Hugging Face Space (Docker)
                                          → rebuild → serve app.py
```

---

## One-time setup

### 1. Create the Space

- huggingface.co → **New Space**
- Owner: your account · Name: `CollegeFootballDashboard` (must match `HF_SPACE`
  in the workflow) · License: MIT
- **SDK: Docker** · Hardware: CPU basic (free) is enough; upgrade CPU if the
  interactive Playoff Simulator feels slow
- Create it **empty** (no template).

### 2. Create a Hugging Face write token

- huggingface.co → Settings → **Access Tokens** → *New token*
- Role: **Write** · copy it.

### 3. Add GitHub repository secrets

`Settings → Secrets and variables → Actions → New repository secret`:

| Secret | Value |
|---|---|
| `HF_TOKEN` | the write token from step 2 (**required**) |
| `CFBD_API_KEY` | your CollegeFootballData key (optional — without it the deploy uses the synthetic demo dataset) |

### 4. Point the workflow at your Space

The workflow defaults to `HF_USERNAME=sammmmmmm25`, `HF_SPACE=CollegeFootballDashboard`.
To change either **without editing the file**, add repo **Variables** (Settings →
Secrets and variables → Actions → *Variables* tab): `HF_USERNAME`, `HF_SPACE`.
The `HF_USERNAME` is your **Hugging Face** username (from your HF profile URL),
which may differ from your GitHub username — a mismatch is the usual cause of a
`git push` "repository not found" (exit 128).

Also update the two GitHub URLs in `deploy/SPACE_README.md` if your repo path
differs.

### 5. Trigger the first deploy

- GitHub → **Actions** → *Deploy to Hugging Face Space* → **Run workflow**, or
- push any change under `src/`, `pages/`, `app.py`, `config/`, `requirements.txt`,
  `Dockerfile`, or `.streamlit/`.

First run takes ~3 min (data pipeline) + a few min for the Space's first Docker
build. After that the Space is live at
`https://huggingface.co/spaces/<user>/<space>`.

---

## How updates work after setup

| Trigger | What happens |
|---|---|
| **Schedule** (every 6 h) | full `update_dashboard.py` → push → Space rebuilds with fresh data |
| **Push to `main`** touching code paths | same as above (keeps deployed artifacts consistent with deployed code) |
| **Manual** (`workflow_dispatch`) | same |
| Docs-only push | nothing (path filter) |

The **freshness badge** on every page reports the truth: *Current* if the last
push landed within the expected interval, *Delayed* if one refresh was missed,
*Stale* after 24 h. If the Action fails, the Space keeps serving the last good
artifacts and shows *Delayed* / *Stale* — it never blanks or crashes.

For **game-day** cadence, uncomment the `*/15` cron line in the workflow.

---

## Notes & limitations

- **Force-push** to the Space keeps its git history flat (one commit), so the
  few MB of artifacts pushed each cycle don't accumulate.
- The Space needs **no `CFBD_API_KEY`** — it only reads artifacts. The key lives
  only in GitHub Actions secrets.
- Each deploy re-pulls **all** configured seasons (`config → season.backtest_seasons`
  + current), ~3 min. Making the fetch incremental is a tracked improvement
  (`docs/LIMITATIONS.md`).
- Rebuildable intermediates (`data/processed/features.parquet`,
  `last_update_report.json`) are excluded from the Space to keep it lean; the app
  rebuilds the feature matrix in-process.
- **Yearly:** bump `current_season` and extend `backtest_seasons` in
  `config/config.yaml` each August.

---

## Other hosts

The `Dockerfile` is host-agnostic (listens on `$PORT`, defaults to 7860):

- **Streamlit Community Cloud** — connect the GitHub repo directly; add
  `CFBD_API_KEY` in its secrets UI. Same ephemeral-FS model as Spaces, so keep
  the GitHub Action pushing artifacts (to a branch it reads, or commit them).
- **Railway / Render / Fly.io** — these give a persistent disk and a real cron,
  so `scripts/update_dashboard.py` can run in-place (see `scripts/crontab.example`)
  instead of the push-from-Actions pattern.
- **Plain Docker:**
  ```bash
  docker build -t cfb-dashboard .
  docker run -p 7860:7860 cfb-dashboard      # http://localhost:7860
  ```
