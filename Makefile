# Convenience commands. `make help` lists them.
PY ?= python

.PHONY: help setup data dataset train evaluate simulate update app test lint all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:        ## Install dependencies into the current environment
	$(PY) -m pip install -r requirements.txt

data:         ## Fetch / update raw data into the local store (idempotent)
	$(PY) scripts/update_data.py

dataset:      ## Build the processed dataset + feature matrix
	$(PY) scripts/build_dataset.py

train:        ## Train Elo + baselines + primary model, save the bundle
	$(PY) scripts/train_models.py

evaluate:     ## Print the saved out-of-sample evaluation
	$(PY) scripts/evaluate_models.py

simulate:     ## Run the Monte Carlo season simulation + leverage
	$(PY) scripts/run_simulation.py --iterations 10000

update:       ## Full pipeline: data -> train -> simulate -> artifacts
	$(PY) scripts/update_dashboard.py --iterations 10000

app:          ## Launch the Streamlit dashboard
	streamlit run app.py

test:         ## Run the test suite
	$(PY) -m pytest

lint:         ## Run ruff
	ruff check .

all: data train simulate ## Data + train + simulate in one go
