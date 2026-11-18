# README

## Input/output contract
- See `artifacts/feature_contract.json`
- Also see `tests/reference_case.json` for an example of expected input and output

## Setup
- Preprocessing: see submission_checklist.md
- Launch Docker daemon
- Run `make build` followed by `make run`. Dependencies are handled in build step.
- (CLI) Use curl to send request to http://127.0.0.1:8000/predict 

## Artifact loading instructions 
- Covered under Setup - Preprocessing

## Commands to reproduce results
- Testing: run `make test` or `pytest test/test_app.py`
- Prediction: run `curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request_tofail.json`

## What would need attention before a real deployment
- Would consider packaging the research database + workflow into a seperate container, as well as a seperate test container; further automate test and prediction model updates in real deployment
- Consider adding GUI for user-friendly way to request prediction

## Submission checklist
- See submission_checklist.md