# Lecture 6: Algorithms, Performance and LLM Memory

Thursday, October 22, 2026 · Advanced Python II · 9 a.m.–noon Pacific.

**Homework:** [HW 3: Advanced Python](../../Homeworks/HW3/README.md), covering Lectures 5 and 6, is due Thursday, November 5, 2026, at 11:59 p.m. Pacific (PST).

## Thursday lesson materials

- [Google Slides](https://docs.google.com/presentation/d/1K1NefuYkRKFRPo0Qvd5NSLWJXOLY4Fc6qEheDTKv9ug/edit): the current lesson deck.
- [PowerPoint slides with speaker notes](thursday-2026-10-22/UCB_MFE_Lesson_6_Algorithms_Performance_LLM_Memory.pptx): all 56 slides, exported from Google Slides after class.
- [Prepared demo notebook](thursday-2026-10-22/Thursday_Prepared_Notebook.ipynb): search, repeated lookup, recursion, generators, bounded async, KV-cache memory and independently checked tool execution. **Outputs and execution counts are cleared** for live teaching.
- [Saved demo results](thursday-2026-10-22/Thursday_Demo_Reader.html): a separate offline reader containing the prepared outputs and explanations.
- [Interactive KV-cache walkthrough](thursday-2026-10-22/KV_Cache_Walkthrough.html): step through prefill/decoding and explore cache memory growth. This is an explanatory simulation, not a model benchmark.

Download the two HTML files and notebook together to use the local links. Open HTML files in a browser; GitHub displays their source rather than running the interactive page. The Google Slides deck contains playable videos and direct YouTube links; PDF or PowerPoint viewers may handle embedded videos differently.

### Running the demo

Open the prepared notebook in Jupyter with the course Python environment and run it from top to bottom. It uses only the Python standard library and requires no API keys, external data, network calls or GPU. Async examples use top-level `await`, as supported by Jupyter, and simulated waits rather than remote requests.

The core route is A3 (generators), B1/B2 (async and failure accounting), C (KV-cache arithmetic), and D (tool execution/checks). A1/A2 and the recursion comparison support the algorithms section. All ten code cells passed in a fresh Jupyter kernel before outputs were cleared. Timings and allocation measurements vary by machine.

### Topics

1. Input size, Big O, time and auxiliary space; search, sorting, recursion and data structures.
2. Generators and retained intermediate data.
3. CPU/GPU execution, capacity, latency, bandwidth and data movement.
4. Concurrency, parallelism, `async`/`await`, in-flight limits and failure accounting.
5. LLM inference, tokens, attention, prefill/decoding, KV-cache reuse and memory costs.
6. Tool execution and independent correctness checks carried forward from Monday.

## Prior-course slides shown in class

- [January 2026 Lesson 6 — exported original deck](january-2026-reference/UCB_MFE_January_2026_Lesson_6.pptx).
- [Original Google Slides](https://docs.google.com/presentation/d/1K1syN_Pu9GI-YYLNtGQF32yIW69nFcnn04as9i3McAo/edit): concurrency begins on slide 25; slides 30–31 compare threading/multiprocessing, 32–33 cover concurrency/parallelism and async, and 34 covers asyncio/semaphores.
- [Prior-course Lesson 5 materials](../Lecture%205/README.md#january-cohort-source-materials), also referenced in the October lessons.

The `january-2026-reference` file is a fresh October 22 export of the original Google deck shown in class. The existing `UCB MFE 2026 - Lesson 6.pptx` and `UCB_MFE_2026_Lesson_6_Demos.ipynb` remain as the earlier imported reference materials. The older notebook has HTTP examples requiring internet access and a multiprocessing example using `lecture6_workers.py`; those requirements do not apply to Thursday's prepared notebook.

These archived decks preserve the original teaching material. For current assumptions and qualifications, use Thursday's deck: Big O is an asymptotic upper bound rather than a synonym for worst case; a concurrency semaphore limits active work rather than requests per second; conventional GIL behavior does not describe every Python build or native-library operation.

## Original lesson videos

- [Big O](https://www.youtube.com/watch?v=g2o22C3CRfU) — Thursday slide 11.
- [Sorting algorithms](https://www.youtube.com/watch?v=RfXt_qHDEPw) — Thursday slide 17.
- [Binary search trees](https://www.youtube.com/watch?v=mtvbVLK5xDQ) — Thursday slide 22.
