"""Sequential, portable evaluation on a selected external manifest."""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from emem_bench.data import contained_path
from .env import EmbodiedMemorizerEnv
from .grounding import (
    bind_action_to_observation_space,
    repair_action_id_from_structured_prediction,
)
from .planner import APIPlanner
from .tool_agent import InvalidPhaseOutput, ToolAgent


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    families = {}
    for family, items in grouped.items():
        families[family] = {
            "episodes": len(items),
            "sr": sum(bool(x["result"]["task_completed"]) for x in items) / len(items),
            "err": sum(float(x["result"]["err"]) for x in items) / len(items),
        }
    return {
        "episodes": len(rows),
        "by_family": families,
        "macro_sr": sum(x["sr"] for x in families.values()) / len(families)
        if families
        else None,
        "macro_err": sum(x["err"] for x in families.values()) / len(families)
        if families
        else None,
    }


def evaluate(rows, args):
    if args.mode not in {"full_context", "emem"}:
        raise ValueError("Supported evaluation modes: full_context, emem")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a new, empty output directory; existing runs are never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    public_args = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"api_key", "handler"}
    }
    (output / "run_config.json").write_text(json.dumps(public_args, indent=2))
    args.auto_memory_observe_step = False
    results = []
    for index, row in enumerate(rows):
        episode_path = contained_path(args.data_root, row["source"]["episode_path"])
        env = EmbodiedMemorizerEnv(
            episode_path=str(episode_path),
            log_path=str(output / "environment" / str(index)),
            max_steps=args.max_steps,
            width=500,
            height=500,
            strict_actions=True,
            include_all_targets=True,
            expose_l2_task_object_label=False,
        )
        matches = [
            i
            for i, item in enumerate(env.eval_items)
            if item["probe"].get("probe_id") == row["source"]["probe_id"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Manifest probe is not unique: {row['id']}")
        env._current_episode_num = matches[0]
        agent = None
        calls = []
        errors = []
        started = False
        infrastructure_failed = False
        try:
            observation = env.reset()
            started = True
            if args.mode == "emem":
                agent = ToolAgent(args, observation)
                agent.ingest(observation.get("context", []), episode_path=episode_path)
                previous = []
                for _ in range(args.max_steps):
                    observation, done, executed, reason = agent.act(
                        env, observation, previous
                    )
                    previous = executed
                    if reason == "simulator_step_error":
                        raise RuntimeError(
                            "Simulator failed while executing the action sequence"
                        )
                    if (
                        done
                        or env._current_step >= args.max_steps
                        or reason == "simulator_step_error"
                    ):
                        break
            else:
                planner = APIPlanner(
                    model_name=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    max_tokens=args.max_tokens,
                    temperature=0,
                    timeout=args.timeout,
                    retries=2,
                    retry_delay=2,
                    text_only=args.text_only,
                    max_context_steps=80,
                    max_visible_objects=40,
                )
                for _ in range(args.max_steps):
                    action, reasoning, payload, raw = planner.act(observation)
                    action, _ = repair_action_id_from_structured_prediction(
                        action, payload, observation
                    )
                    action, _ = bind_action_to_observation_space(action, observation)
                    observation, _, done, info = env.step(action, reasoning=reasoning)
                    calls.append(planner.update_info(info, action, payload, raw))
                    if done:
                        break
        except InvalidPhaseOutput as exc:
            errors.append(str(exc))
        except Exception as exc:
            # An infrastructure error aborts the run; it is not scored as a model failure.
            infrastructure_failed = True
            (output / "infrastructure_error.json").write_text(
                json.dumps(
                    {
                        "id": row["id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
            )
            raise
        finally:
            try:
                if started and env.evaluator is not None and env.probe is not None:
                    result = env.current_result()
                    entry = {
                        "id": row["id"],
                        "family": row["family"],
                        "result": asdict(result),
                        "errors": errors,
                        "infrastructure_failed": infrastructure_failed,
                        "calls": agent.calls if agent else calls,
                    }
                    (output / f"episode_{index:04d}.json").write_text(
                        json.dumps(entry, indent=2)
                    )
                    env.save_episode_log()
                    results.append(entry)
            finally:
                env.close()
        summary = summarize(results)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(rows),
                    "macro_sr": summary["macro_sr"],
                }
            ),
            flush=True,
        )
    return summarize(results)
