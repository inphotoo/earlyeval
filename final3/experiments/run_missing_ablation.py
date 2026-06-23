from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from final3.experiments.rq_final_ablation import _summarize


DEFAULT_ROOT = Path("paper/experiments/rq_final_lightgbm_17")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially resume missing SWEVerify ablation folds from existing command_index.csv files."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--run-subdirs",
        nargs="+",
        default=[
            "sweverify_ablation_default_reg_full16",
            "sweverify_ablation_balanced4",
            "sweverify_ablation_fine_grained_full16",
        ],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--vmem-kb", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--refresh-reporting-detail", action="store_true")
    return parser.parse_args()


def _apply_limits(vmem_kb: int) -> None:
    if vmem_kb > 0:
        limit = int(vmem_kb) * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _command_with_thread_cap(command: list[str], threads: int) -> list[str]:
    capped: list[str] = []
    skip_next = False
    for idx, item in enumerate(command):
        if skip_next:
            skip_next = False
            continue
        if item == "--max-cpu-threads" and idx + 1 < len(command):
            capped.extend([item, str(int(threads))])
            skip_next = True
        else:
            capped.append(item)
    return capped


def _load_jobs(root: Path, run_subdirs: list[str], *, force: bool, threads: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    base = root / "ablations" / "sweverify"
    for run_subdir in run_subdirs:
        run_dir = base / run_subdir
        index_path = run_dir / "command_index.csv"
        if not index_path.exists():
            skipped.append({"run_subdir": run_subdir, "status": "missing_command_index", "path": str(index_path)})
            continue
        frame = pd.read_csv(index_path)
        for _, row in frame.iterrows():
            output_dir = Path(str(row["output_dir"]))
            marker = output_dir / "safe_stop_test_selected.csv"
            success = output_dir / "_SUCCESS"
            item = {
                "run_subdir": run_subdir,
                "profile": str(row["profile"]),
                "fold_id": str(row["fold_id"]),
                "test_model": str(row["test_model"]),
                "output_dir": str(output_dir),
                "log": str(row["log"]),
            }
            if marker.exists() and not force:
                if not success.exists():
                    success.write_text("completed from existing marker\n", encoding="utf-8")
                skipped.append({**item, "status": "existing"})
                continue
            command = str(row["command"]).split()
            jobs.append({**item, "command": _command_with_thread_cap(command, threads), "status": "pending"})
    return jobs, skipped


def _run_job(job: dict[str, Any], *, threads: int, vmem_kb: int) -> dict[str, Any]:
    out_dir = Path(str(job["output_dir"]))
    log_path = Path(str(job["log"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in [
        "SWE_MAX_CPU_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[key] = str(int(threads))
    env["MALLOC_ARENA_MAX"] = env.get("MALLOC_ARENA_MAX", "2")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(job["command"]) + "\n\n")
        log.flush()
        proc = subprocess.run(
            job["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=(lambda: _apply_limits(vmem_kb)),
        )
    result = {key: value for key, value in job.items() if key != "command"}
    result["returncode"] = int(proc.returncode)
    result["status"] = "completed" if proc.returncode == 0 else "failed"
    if proc.returncode == 0:
        (out_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
    return result


def _refresh_summaries(root: Path, run_subdirs: list[str]) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    base = root / "ablations" / "sweverify"
    for run_subdir in run_subdirs:
        run_dir = base / run_subdir
        if run_dir.exists():
            refreshed.append({"run_subdir": run_subdir, "summary": _summarize(run_dir)})
    return refreshed


def main() -> int:
    args = parse_args()
    jobs, skipped = _load_jobs(args.root, args.run_subdirs, force=bool(args.force), threads=int(args.threads))
    if args.max_jobs is not None:
        jobs = jobs[: max(0, int(args.max_jobs))]
    payload: dict[str, Any] = {
        "root": str(args.root),
        "execute": bool(args.execute),
        "threads": int(args.threads),
        "vmem_kb": int(args.vmem_kb),
        "planned_jobs": len(jobs),
        "pending_jobs": [{key: value for key, value in job.items() if key != "command"} for job in jobs],
        "skipped": skipped,
        "results": [],
    }
    if args.execute:
        for idx, job in enumerate(jobs, start=1):
            print(
                f"[missing-ablation] {idx}/{len(jobs)} {job['run_subdir']} "
                f"{job['profile']}/{job['fold_id']}",
                flush=True,
            )
            result = _run_job(job, threads=int(args.threads), vmem_kb=int(args.vmem_kb))
            payload["results"].append(result)
            if result["status"] != "completed":
                print(f"[missing-ablation] failed: {result}", flush=True)
        payload["summaries"] = _refresh_summaries(args.root, args.run_subdirs)
        if args.refresh_reporting_detail:
            proc = subprocess.run([sys.executable, "-m", "final3.experiments.build_reporting_detail"])
            payload["reporting_detail_returncode"] = int(proc.returncode)
    else:
        payload["note"] = "dry-run only; add --execute to run missing folds sequentially."
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(item.get("status") == "failed" for item in payload["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
