# Lesson 9 — Systems architecture and production engineering

November 2, 2026 · UCB MFE

- [Google Slides deck](https://docs.google.com/presentation/d/1LKEAP9jQ-zxPsbADbvQEaMCamn06ZXEQzDXL5xLNSqM/edit) — 44 teaching slides plus six references; speaker notes included.
- [PowerPoint slides](UCB_MFE_Lesson_9_Systems_Architecture.pptx) — exported from Google Slides after class, with speaker notes.
- [Prepared notebook](Lesson_9_Demos.ipynb) — cleared outputs.
- [Verified demo reader](Lesson_9_Demo_Reader.html) — saved outputs; works without the API.
- [PDF deck](UCB_MFE_Lesson_9_Systems_Architecture.pdf) — exported from native Google Slides.

Download the notebook, demo folder, and HTML reader together. Open the HTML file locally in a browser; GitHub displays its source. Videos require the linked online player; PDF and PowerPoint viewers may handle embedded media differently. The downloadable slides are also available if you cannot access the Google Slides link.

**Homework:** [HW5: Serve and containerize a model](../../Homeworks/HW5/hw_5.md), due **Monday, November 16, 2026, at 11:59 p.m. Pacific (PST)**.

## Reading and video from class

- [Anthropic: A postmortem of three recent issues](https://www.anthropic.com/research/a-postmortem-of-three-recent-issues) (September 17, 2025) — infrastructure failures, diagnosis, and gaps in quality monitoring. Relate the incident to validation, deployment changes, and production observation.
- [Mastering Chaos — A Netflix Guide to Microservices](https://www.youtube.com/watch?v=CZ3wIuvmHeM) (InfoQ) — the Netflix talk shown in class and embedded in reference slide A5. Discuss service boundaries, failure isolation, and the operational costs of distributed systems.

## Prior-course reference slides

- [January 2026 Lesson 9 — PowerPoint](january-2026-reference/UCB_MFE_January_2026_Lesson_9.pptx)
- [January 2026 Lesson 9 — PDF](january-2026-reference/UCB_MFE_January_2026_Lesson_9.pdf)
- [Original Google Slides](https://docs.google.com/presentation/d/1A3JsobOUMWfKrIaK0016ufD43HaMf3t_Xe10Df7kY3s/edit)

These 51-slide reference files preserve the previous course's deck and were exported on November 2, 2026. Use this course's deck and HW5 instructions for current requirements and tooling.

## Run the two core demos

From this folder:

```sh
cd demo
uv sync --locked
uv run --locked jupyter lab ../Lesson_9_Demos.ipynb
```

On this Mac, `Launch_Demos.command` opens the notebook. Run Setup before class; it starts a separate localhost API process. Stop it using the notebook’s cleanup cell after class. All example data are synthetic and included; the saved pipeline is already fitted.

Demo A at slide 14: the intentional basis-point bug fails, then the corrected implementation passes the same test. Demo B at slide 26: direct and API predictions agree, then invalid input is rejected. Docker at slide 36 is optional.

## Project checks

From demo/:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

The deliberately broken function is only selected with `--fee-version buggy`. It is not used by the service or default checks. The known fee test’s expected result is independently specified.

## Container

```sh
docker build -t mfe-lesson9:classroom-v1 .
docker run --rm --name mfe-lesson9-demo -p 127.0.0.1:8770:8000 mfe-lesson9:classroom-v1
```

Repeat the notebook API request at `http://127.0.0.1:8770/predict`. Stop with Ctrl-C. The requirements file is exported from uv.lock using `uv export --locked --no-dev --no-hashes --no-emit-project -o requirements.txt`.

## Model contract

`momentum_20d` is the trailing 20-day return as a fraction, bounded from -1 to 1. `volatility_20d` is annualized volatility estimated from 20 days, as a fraction, bounded from 0 to 2. Bounds are classroom choices. Fields are required, finite and strict numeric types; extra fields are rejected. The prediction is a return fraction with model version `classroom-v1`.

The synthetic training target is `0.4 * momentum_20d - 0.1 * volatility_20d + 0.002`. This model demonstrates mechanics and has no claim to predictive quality. To regenerate the trusted artifact, run `uv run --locked python train_model.py`. Do not load untrusted pickle/joblib files.

## CI and provenance

The [lesson workflow](../../.github/workflows/lesson-9.yml) runs these checks for changes to this demo. Required status checks need separate repository configuration.

Adapted from the January 2025/2026 Lesson 9 deck and January 2026 HW5 scaffold. Original decks are unchanged. Additional details and primary references are in the instructor guide and slide notes.
