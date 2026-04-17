# Audio transcriber (faster-whisper)

Small CLI that transcribes local audio files (including M4A) to a UTF-8 text file using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No API keys; the first run downloads the selected model.

Optional **`summarize.py`** reads a transcript `.txt` and produces **Markdown meeting minutes** using **[Ollama](https://ollama.com)** on your machine (offline after the model is pulled). It uses only the Python standard library for HTTP—no extra pip packages for summarization.

**Repository layout:** clone this repo, `cd` into it, then follow Setup. All example commands assume your shell’s **current working directory is the repository root** (the folder that contains `transcribe.py` and `requirements.txt`).

## Prerequisites

- **Python** 3.10 through 3.13 (stable releases). Use an interpreter that has PyPI wheels for `ctranslate2` and `onnxruntime` (e.g. `python3.12`) if your default `python3` is a very new or pre-release version without wheels.
- **ffmpeg** on your `PATH` (required to decode M4A and most formats)

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg
sudo apt install -y python3.12 python3.12-venv python3.12-full
```

Use `python3.11` / `python3.11-venv` / `python3.11-full` (or 3.10) instead if 3.12 packages are not in your distro’s repositories.

### Optional: build Python from source (e.g. pyenv)

You do **not** need this if you use your distribution’s `python3.12` (or 3.11 / 3.10) package. These are typical build dependencies when compiling Python yourself:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl git \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

On some releases the ncurses dev package may be named `libncurses-dev` instead of `libncursesw5-dev`; install the one your distro provides.

## Setup

Always create the virtualenv with **Python 3.12** (or 3.11 / 3.10)—not your default `python3` if it is **3.14+** or a pre-release. Newer versions often lack wheels for `ctranslate2`, `onnxruntime`, and `av`, and Debian/Ubuntu’s **system** `pip` can error while installing wheels (e.g. `TypeError: Can't instantiate abstract class WheelDistribution…`).

**Recommended (no activation ambiguity):** use the venv’s interpreter path for every step.

```bash
rm -rf .venv
python3.12 -m venv .venv   # needs python3.12-venv; or python3.11 / python3.10
```

Confirm pip lives **inside** the venv (not `/usr/lib/python3/dist-packages/pip`):

```bash
.venv/bin/python -m pip -V
# expect: pip … from …/<repo>/.venv/lib/python3.12/site-packages/pip
```

If that command fails or still shows system pip, bootstrap pip into the venv:

```bash
.venv/bin/python -m ensurepip --upgrade
```

Install dependencies and run the script:

```bash
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python transcribe.py "path/to/your/audio.m4a"
```

**If you prefer `source .venv/bin/activate`:** run `which python`, `python -V`, and `python -m pip -V` after activating. If `python -m pip -V` mentions `/usr/lib/python3/dist-packages`, your shell is not using the venv’s `python` (or pip was never installed into the venv)—use the `.venv/bin/python …` commands above instead.

If `pip` tries to build **PyAV** from source and fails with `pkg-config is required`, use Python 3.10–3.13 so a binary `av` wheel installs, or install FFmpeg dev headers for your OS.

## Usage

If your shell’s `python` is **3.14+** (e.g. `/usr/bin/python3.15`), use the venv interpreter explicitly—**do not rely on** `python transcribe.py` unless `which python` points at `.venv/bin/python`.

```bash
.venv/bin/python transcribe.py "path/to/your/audio.m4a"
```

After a correct `source .venv/bin/activate`, plain `python transcribe.py …` is fine too.

By default this writes a **timestamped** file next to the audio, e.g. `meeting_20260406-153045.txt`, so reruns do not overwrite the previous transcript. Use **`--no-timestamp`** for a fixed `<stem>.txt` (overwrites on rerun).

Choose the output path explicitly:

```bash
.venv/bin/python transcribe.py "path/to/your/audio.m4a" -o transcript.txt
```

Optional formatting and incremental saves:

```bash
.venv/bin/python transcribe.py "path/to/your/audio.m4a" --pause-breaks --flush-minutes 1
```

`--pause-breaks` inserts a blank line when the gap between two transcribed segments is at least **`--pause-gap`** seconds (default `1.0`). `--flush-minutes N` appends new text to the output every *N* minutes while the job runs (`0`, the default, means **write only when finished**).

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output path. If omitted, default is `<stem>_YYYYMMDD-HHMMSS.txt` beside the input. |
| `--no-timestamp` | Default output is `<stem>.txt` instead of a timestamped name (overwrites on rerun). |
| `--pause-breaks` | Blank line between segments when silence between them is ≥ `--pause-gap`. |
| `--pause-gap` | Seconds of gap required for a pause break (with `--pause-breaks`). Default: `1.0`. |
| `--flush-minutes` | If positive, append progress to the output every *M* minutes; `0` = single write at the end (default). |
| `-m`, `--model` | Model size: `tiny`, `base` (default), `small`, `medium`, `large-v2`, `large-v3`, etc. Larger models are more accurate but slower and bigger to download. |
| `-l`, `--language` | Force language (e.g. `en`). Omit to auto-detect. |
| `--device` | `cpu` (default) or `cuda` if you have a GPU set up. |
| `--compute-type` | e.g. `int8` on CPU, `float16` with `--device cuda`. Default lets the library choose. |

### GPU example

```bash
.venv/bin/python transcribe.py "path/to/your/audio.m4a" --device cuda --compute-type float16
```

The first run downloads the model into your user cache; download size depends on `--model`.

## Meeting minutes (offline Ollama)

1. Install [Ollama](https://ollama.com).

   If `ollama` is missing on Ubuntu/Debian, use one of these:

   ```bash
   sudo snap install ollama
   # then either log out/in, or open a new terminal so `ollama` is on PATH
   ```

   or the official installer:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Start Ollama and pull a chat model once (downloads several GB, like Whisper weights):

   ```bash
   ollama serve
   # in another terminal:
   ollama pull llama3.2
   ```

3. Run the summarizer on a transcript:

   ```bash
   .venv/bin/python summarize.py "transcript.txt"
   ```

   Default output: `transcript_minutes_YYYYMMDD-HHMMSS.md` next to the transcript. Use `-o path.md` to set the file explicitly.

Long transcripts are split into chunks (**map** → bullets per chunk, then **reduce** → one minutes document). Tune with `--chunk-chars` (default `14000`) and `--chunk-overlap` (default `400`). Use `--model` if you pulled something other than `llama3.2`, and `--base-url` if Ollama is not on `http://127.0.0.1:11434`.

## Troubleshooting

- **`ffmpeg not found`**: Install ffmpeg and ensure it is on `PATH`.
- **`Python X.Y is not supported` / `This process: /usr/bin/python3.15`**: You ran `transcribe.py` with **system Python**, not the venv. Use **`.venv/bin/python transcribe.py …`** from the project directory (see [Usage](#usage)).
- **`could not import faster-whisper` / wrong `sys.executable`**: Packages were installed for a different interpreter. Run `.venv/bin/python -m pip install -r requirements.txt` and `.venv/bin/python transcribe.py …`.
- **`Defaulting to user installation` or `WheelDistribution` / `locate_file`**: `python -m pip` is using **Debian’s system pip**, not the venv. Run `.venv/bin/python -m pip -V`; if it does not show `.venv/lib/…/site-packages/pip`, run `.venv/bin/python -m ensurepip --upgrade`, then install again with `.venv/bin/python -m pip`.
- **`ResolutionImpossible` for `ctranslate2`**: Your install command was not using **Python 3.12** (e.g. 3.15 has no `ctranslate2` wheel). Use `.venv/bin/python -V` → 3.12.x and reinstall with `.venv/bin/python -m pip install -r requirements.txt`.
- **Pip downloads `av-*.tar.gz` and fails**: Prefer Python 3.12 + upgraded pip in the venv; avoid forcing an ancient `faster-whisper` that pulls old `av` sdists.
- **`Could not reach Ollama` / `Connection refused`**: Nothing is listening on **port 11434**. Install Ollama from [ollama.com/download](https://ollama.com/download), then start it: run **`ollama serve`** in a separate terminal (leave it running), or try **`systemctl --user start ollama`** if your package installed a user service. Verify with **`curl -s http://127.0.0.1:11434/api/tags`**, then **`ollama pull llama3.2`** (or whatever you pass to **`--model`**). If the API is on another host/port, use **`summarize.py --base-url …`**.
