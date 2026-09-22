"""Small public CLI. Heavy simulator and Hub dependencies are loaded on demand."""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .data import FAMILIES, contained_path, load_manifest


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def selection(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--per-family", type=positive, default=1)
    group.add_argument(
        "--all", action="store_true", help="Select every matching manifest row"
    )


def audit(rows, args):
    from .evaluation.evaluator import EpisodeEvaluator
    from .evaluation.memory_adapter import MemoryRuntime

    outputs = []
    for row in rows:
        path = contained_path(args.data_root, row["source"]["episode_path"])
        evaluator = EpisodeEvaluator(str(path))
        probe = next(
            p
            for p in evaluator.get_probes()
            if p["probe_id"] == row["source"]["probe_id"]
        )
        contexts = evaluator.get_context_for_model()
        runtime = MemoryRuntime(
            contexts,
            probe["instruction"],
            current_scene=row["source"].get("probe_scene"),
            current_namespace=evaluator.episode_id,
            embedding_model="lexical-hash",
        )
        memory_context, stats = runtime.build_context()
        outputs.append(
            {
                "id": row["id"],
                "family": row["family"],
                "history_steps": sum(len(s["steps"]) for s in contexts),
                "stats": stats,
                "retrieved_context": memory_context,
            }
        )
    return {
        "scope": "Offline ingestion/retrieval diagnostic, not model task success",
        "episodes": outputs,
    }


def construct(args):
    from .construction.episodes import (
        generate_l2_dynamic_tracking,
        generate_l2_failure_retrieval,
        generate_l2_passive_retrieval,
        generate_l3_by_rule,
    )
    from .construction.utils import save_episode_to_dir

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Construction requires a new, empty output directory")
    if args.family == "l3_owner_habit":
        if not args.rule or not args.failure_scenes:
            raise ValueError("EG construction requires --rule and --failure-scenes")
        from .construction.experience_templates import ConstraintRule, PropertyFilter

        rule_dict = json.loads(args.rule.read_text())
        for name in ["object_filter", "target_filter", "preferred_target_filter"]:
            if rule_dict.get(name):
                rule_dict[name] = PropertyFilter(**rule_dict[name])
        rule = ConstraintRule(**rule_dict)
        episode = generate_l3_by_rule(
            {rule.rule_id: rule},
            str(args.output),
            rule.rule_id,
            args.failure_scenes,
            macro_scene=args.scene,
            seed=args.seed,
        )
    else:
        generate = {
            "l2_dynamic": generate_l2_dynamic_tracking,
            "l2_interaction": generate_l2_failure_retrieval,
            "l2_passive": generate_l2_passive_retrieval,
        }[args.family]
        episode = generate(
            str(args.output),
            context_scene=args.scene,
            noise_scene=args.noise_scene,
            seed=args.seed,
        )
    if episode is None:
        raise RuntimeError("No valid task could be generated with these scenes/rule")
    path = save_episode_to_dir(episode, str(args.output))
    return {
        "episode_path": path,
        "note": "Generated candidate; audit before adding it to a benchmark release",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EMem-Bench code-only research release"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser(
        "inspect", help="Validate external manifest and report task counts"
    )
    selection(inspect)
    offline = subs.add_parser(
        "audit-memory", help="Offline real-history ingestion/retrieval check"
    )
    selection(offline)
    offline.add_argument("--output", type=Path)
    run = subs.add_parser("evaluate", help="Run a model through the actual simulator")
    selection(run)
    run.add_argument(
        "--mode", choices=["full_context", "emem"], default="full_context"
    )
    run.add_argument(
        "--model", required=True, help="Served base model or trained checkpoint alias"
    )
    run.add_argument(
        "--base-url", required=True, help="Chat-completions-compatible endpoint"
    )
    run.add_argument("--api-key-env", default="EMEM_API_KEY")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--embedding-model", default="lexical-hash")
    run.add_argument("--max-steps", type=positive, default=8)
    run.add_argument("--max-tokens", type=positive, default=4096)
    run.add_argument("--timeout", type=positive, default=120)
    run.add_argument(
        "--text-only", action="store_true", help="Diagnostic only: omit RGB inputs"
    )
    score = subs.add_parser(
        "score", help="Offline scoring of an already executed action trace"
    )
    score.add_argument("--episode", type=Path, required=True)
    score.add_argument("--probe-id", required=True)
    score.add_argument("--trajectory", type=Path, required=True)
    build = subs.add_parser(
        "construct", help="Generate one candidate task using the original generators"
    )
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--family", choices=FAMILIES, required=True)
    build.add_argument("--scene", required=True)
    build.add_argument("--noise-scene", default="FloorPlan201")
    build.add_argument("--failure-scenes", nargs="+")
    build.add_argument("--rule", type=Path)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output", type=Path, required=True)
    hub = subs.add_parser(
        "download", help="Download a benchmark dataset or model snapshot"
    )
    hub.add_argument("--repo-id", required=True)
    hub.add_argument("--repo-type", choices=["dataset", "model"], required=True)
    hub.add_argument(
        "--revision", required=True, help="Prefer an immutable commit revision"
    )
    hub.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "data_root"):
        args.data_root = args.data_root.expanduser().resolve()
        os.environ["EMEM_DATA_ROOT"] = str(args.data_root)
    if args.command in {"inspect", "audit-memory", "evaluate"}:
        rows = load_manifest(
            args.manifest,
            args.data_root,
            per_family=None if args.all else args.per_family,
            families=args.families,
        )
        if args.command == "inspect":
            result = {
                "selected": len(rows),
                "by_family": dict(Counter(r["family"] for r in rows)),
            }
        elif args.command == "audit-memory":
            if args.output and args.output.exists():
                raise FileExistsError(args.output)
            result = audit(rows, args)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    json.dump(result, stream, indent=2)
                result = {"episodes_checked": len(rows), "output": str(args.output)}
        else:
            args.api_key = os.environ.get(args.api_key_env)
            from .evaluation.run import evaluate

            result = evaluate(rows, args)
    elif args.command == "score":
        from .evaluation.evaluator import EpisodeEvaluator

        evaluator = EpisodeEvaluator(str(args.episode))
        probe = next(
            p for p in evaluator.get_probes() if p["probe_id"] == args.probe_id
        )
        trace = json.loads(args.trajectory.read_text())
        actions = trace if isinstance(trace, list) else trace["model_actions"]
        result = asdict(evaluator.evaluate_probe(probe, actions))
    elif args.command == "construct":
        result = construct(args)
    else:
        from huggingface_hub import snapshot_download

        if args.local_dir.exists() and any(args.local_dir.iterdir()):
            raise FileExistsError(
                "Download into a new directory to preserve existing artifacts"
            )
        destination = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=str(args.local_dir),
        )
        result = {"local_dir": destination}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
