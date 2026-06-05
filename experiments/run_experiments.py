#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import os
import re
import subprocess
import sys
import time

try:
    import yaml
except ImportError as exc:
    print("Missing dependency: PyYAML. Install with `pip install pyyaml`.", file=sys.stderr)
    raise SystemExit(1) from exc


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _resolve_config_ref(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _load_config(path: str, seen: set[str] | None = None) -> dict:
    path = os.path.abspath(path)
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance detected at {path}")
    seen.add(path)

    raw = _load_yaml(path)
    base_dir = os.path.dirname(path)
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]

    config: dict = {}
    for parent in extends:
        config = _merge(config, _load_config(_resolve_config_ref(parent, base_dir), seen))

    for key in ("models", "architectures"):
        if raw.get(key):
            raw[key] = _resolve_config_ref(str(raw[key]), base_dir)

    seen.remove(path)
    return _merge(config, raw)


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _expand_env_defaults(value: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        env_value = os.environ.get(match.group(1))
        return env_value if env_value not in (None, "") else match.group(2)

    return _ENV_DEFAULT_RE.sub(_replace, value)


def _expand_strings(value):
    if isinstance(value, dict):
        return {key: _expand_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_strings(item) for item in value]
    if isinstance(value, str):
        value = _expand_env_defaults(value)
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _to_cli_args(args_dict: dict) -> list[str]:
    cli: list[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            cli.append(flag)
            cli.extend([str(item) for item in value])
            continue
        cli.extend([flag, str(value)])
    return cli


def _flatten_config(config: dict) -> dict:
    flattened: dict = {}
    section_keys = {"model", "data", "train", "eval", "run"}

    def _walk(prefix: str, obj: dict) -> None:
        for key, value in obj.items():
            if isinstance(value, dict):
                if key in section_keys and not prefix:
                    _walk("", value)
                else:
                    _walk(f"{prefix}{key}_", value)
            else:
                flattened[f"{prefix}{key}"] = value

    _walk("", config)
    return flattened


def _git_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return ""


def _write_registry_row(path: str, row: dict) -> None:
    header = [
        "timestamp",
        "run_name",
        "status",
        "exit_code",
        "git_hash",
        "config_path",
        "log_path",
    ]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if not exists:
            handle.write(",".join(header) + "\n")
        values = [str(row.get(col, "")) for col in header]
        handle.write(",".join(values) + "\n")


def _merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_ref(name: str, presets: dict, key: str) -> dict:
    if not name:
        return {}
    if name not in presets:
        raise ValueError(f"Unknown {key} preset: {name}")
    return presets[name]


def _apply_model_arch(
    config: dict,
    model_defs: dict,
    arch_defs: dict,
) -> dict:
    updated = copy.deepcopy(config)
    arch_name = updated.get("arch")
    if arch_name:
        arch_preset = _resolve_ref(arch_name, arch_defs, "arch")
        updated = _merge(updated, arch_preset)
        updated["arch"] = arch_name
    enc_name = updated.get("encoder")
    dec_name = updated.get("decoder")
    if enc_name or dec_name:
        enc_defs = model_defs.get("encoders", {})
        dec_defs = model_defs.get("decoders", {})
        encoder_preset = _resolve_ref(enc_name, enc_defs, "encoder")
        decoder_preset = _resolve_ref(dec_name, dec_defs, "decoder")
        updated = _merge(updated, encoder_preset)
        updated = _merge(updated, decoder_preset)
        updated.pop("encoder", None)
        updated.pop("decoder", None)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment sweeps from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="")
    args = parser.parse_args()

    config = _load_config(args.config)
    defaults = config.get("defaults", {})
    flat_defaults = _flatten_config(defaults)
    experiments = config.get("experiments", [])
    models_path = config.get("models")
    model_defs = _load_yaml(models_path) if models_path else {}
    arch_defs = {}
    arch_path = config.get("architectures")
    if arch_path:
        arch_defs = _load_yaml(arch_path).get("architectures", {})
    if not experiments:
        experiments = [{"name": flat_defaults.get("run_name", "experiment"), "overrides": {}}]

    git_hash = _git_hash()
    registry_path = os.path.join(flat_defaults.get("results_dir", "results"), "registry.csv")

    gpu_list: list[str] = []
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]

    pending = []
    for exp in experiments:
        name = exp.get("name") or exp.get("run_name")
        overrides = exp.get("overrides", exp.get("args", {}))
        merged = _merge(defaults, {})
        merged = _apply_model_arch(merged, model_defs, arch_defs)
        merged = _merge(merged, overrides)
        merged = _apply_model_arch(merged, model_defs, arch_defs)
        merged.pop("models", None)
        merged.pop("architectures", None)
        merged.pop("arch", None)
        merged = _expand_strings(merged)
        merged = _flatten_config(merged)
        merged = {key: value for key, value in merged.items() if value is not None}
        if name:
            merged["run_name"] = name
        if "results_dir" not in merged:
            merged["results_dir"] = defaults.get("results_dir", "results")
        results_dir = os.path.join(merged["results_dir"], merged.get("run_name", "run"))
        os.makedirs(results_dir, exist_ok=True)
        config_path = os.path.join(results_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=True)

        cmd = [sys.executable, "-u", "main.py"] + _to_cli_args(merged)
        log_path = os.path.join(results_dir, "stdout.log")
        pending.append((merged, cmd, config_path, log_path))

    active: list[dict] = []
    next_gpu = 0

    def _start_job(item):
        nonlocal next_gpu
        merged, cmd, config_path, log_path = item
        if args.dry_run:
            print(" ".join(cmd))
            return None
        env = os.environ.copy()
        if gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[next_gpu % len(gpu_list)]
            next_gpu += 1
        start_ts = dt.datetime.utcnow().isoformat()
        print(f"[run] {merged.get('run_name', '')} -> {log_path}")
        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        return {
            "process": process,
            "log_handle": log_handle,
            "start_ts": start_ts,
            "merged": merged,
            "config_path": config_path,
            "log_path": log_path,
        }

    while pending or active:
        while pending and (args.max_parallel <= 0 or len(active) < args.max_parallel):
            job = _start_job(pending.pop(0))
            if job is not None:
                active.append(job)
        still_active = []
        for job in active:
            process = job["process"]
            ret = process.poll()
            if ret is None:
                still_active.append(job)
                continue
            job["log_handle"].close()
            status = "ok" if ret == 0 else "error"
            print(f"[done] {job['merged'].get('run_name', '')} status={status} exit={ret}")
            _write_registry_row(
                registry_path,
                {
                    "timestamp": job["start_ts"],
                    "run_name": job["merged"].get("run_name", ""),
                    "status": status,
                    "exit_code": ret,
                    "git_hash": git_hash,
                    "config_path": job["config_path"],
                    "log_path": job["log_path"],
                },
            )
        active = still_active
        if active:
            time.sleep(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
