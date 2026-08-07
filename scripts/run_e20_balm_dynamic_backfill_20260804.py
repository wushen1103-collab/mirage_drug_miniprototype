from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/home/test/wsk/mirage_drug_miniprototype")
OUT_ROOT = ROOT / "outputs" / "revision_e20_balm_strict_20260804"
LOG = ROOT / "outputs" / "revision_e20_balm_strict_20260804_backfill.log"
FRAME = "outputs/chembl_assay_primary_20260729/benchmark_frame.csv"
BALM_REPO = "/home/test/ext_balm"
PROTEIN_MODEL = "/home/test/wsk/hf_models/esm2_t30_150m_ur50d"
DRUG_MODEL = "/home/test/wsk/hf_models/chemberta_77m_mtr"
GPU_LIST = list(range(8))
MIN_FREE_MIB = 4500
POLL_SECONDS = 45

TASKS = [
    (split, seed)
    for split in ["assay_cold", "target_cold", "temporal"]
    for seed in [42, 43, 44, 45, 46]
]


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%F %T")
    line = f"===== {stamp} {message} ====="
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""


def gpu_info() -> dict[int, dict[str, str | int]]:
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    out: dict[int, dict[str, str | int]] = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        idx = int(parts[0])
        out[idx] = {"uuid": parts[1], "free_mib": int(float(parts[2]))}
    return out


def compute_apps() -> dict[int, str]:
    text = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    out: dict[int, str] = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            out[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return out


def balm_processes() -> dict[int, str]:
    text = run_text(["pgrep", "-af", "run_external_balm_strict_from_frame.py"])
    out: dict[int, str] = {}
    for line in text.splitlines():
        if "pgrep -af" in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1]
    return out


def running_outputs() -> set[str]:
    outputs: set[str] = set()
    pattern = re.compile(r"--output-dir\s+(\S+)")
    for cmd in balm_processes().values():
        match = pattern.search(cmd)
        if match:
            outputs.add(Path(match.group(1)).name)
    return outputs


def balm_gpu_indices() -> set[int]:
    info = gpu_info()
    uuid_to_idx = {str(v["uuid"]): idx for idx, v in info.items()}
    apps = compute_apps()
    balm_pids = set(balm_processes().keys())
    busy: set[int] = set()
    for pid, uuid in apps.items():
        if pid in balm_pids and uuid in uuid_to_idx:
            busy.add(uuid_to_idx[uuid])
    return busy


def out_dir(split: str, seed: int) -> Path:
    return OUT_ROOT / f"balm_projection_{split}_s{seed}"


def is_done(split: str, seed: int) -> bool:
    return (out_dir(split, seed) / "external_metrics.json").is_file()


def available_gpus() -> list[int]:
    info = gpu_info()
    balm_busy = balm_gpu_indices()
    available = []
    for idx in GPU_LIST:
        if idx in balm_busy:
            continue
        if idx not in info:
            continue
        if int(info[idx]["free_mib"]) < MIN_FREE_MIB:
            continue
        available.append(idx)
    return available


def launch(split: str, seed: int, gpu: int) -> subprocess.Popen:
    out = out_dir(split, seed)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "./.venv/bin/python",
        "scripts/run_external_balm_strict_from_frame.py",
        "--balm-repo",
        BALM_REPO,
        "--benchmark-frame",
        FRAME,
        "--split-method",
        split,
        "--seed",
        str(seed),
        "--output-dir",
        str(out.relative_to(ROOT)),
        "--model-variant",
        "projection",
        "--protein-model",
        PROTEIN_MODEL,
        "--drug-model",
        DRUG_MODEL,
        "--protein-max-length",
        "1024",
        "--drug-max-length",
        "512",
        "--epochs",
        "30",
        "--patience",
        "6",
        "--batch-size",
        "4",
        "--gradient-accumulation-steps",
        "16",
    ]
    handle = LOG.open("a", encoding="utf-8")
    log(f"BACKFILL START BALM strict {split} seed={seed} gpu={gpu}")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)


def main() -> None:
    os.chdir(ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    active: dict[tuple[str, int], tuple[subprocess.Popen, int]] = {}
    attempts: dict[tuple[str, int], int] = {}
    log("BACKFILL SCHEDULER START")
    while True:
        for task, (proc, gpu) in list(active.items()):
            status = proc.poll()
            if status is None:
                continue
            split, seed = task
            if is_done(split, seed):
                log(f"BACKFILL DONE BALM strict {split} seed={seed} gpu={gpu} status={status}")
                del active[task]
            else:
                log(f"BACKFILL EXIT WITHOUT METRICS {split} seed={seed} gpu={gpu} status={status}")
                del active[task]
                attempts[task] = attempts.get(task, 0) + 1

        completed = {task for task in TASKS if is_done(*task)}
        running_names = running_outputs()
        pending = []
        for task in TASKS:
            split, seed = task
            name = out_dir(split, seed).name
            if task in completed or task in active or name in running_names:
                continue
            if attempts.get(task, 0) >= 2:
                continue
            pending.append(task)

        if len(completed) == len(TASKS):
            log("BACKFILL ALL BALM STRICT RUNS FINISHED")
            subprocess.run(["./.venv/bin/python", "scripts/summarize_e20_balm_strict.py"], cwd=ROOT)
            return

        for gpu in available_gpus():
            if not pending:
                break
            task = pending.pop(0)
            active[task] = (launch(task[0], task[1], gpu), gpu)

        done_count = len(completed)
        running_count = len(active) + len(running_names)
        log(f"BACKFILL STATUS done={done_count}/15 running_known={running_count} pending={len(pending)}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

