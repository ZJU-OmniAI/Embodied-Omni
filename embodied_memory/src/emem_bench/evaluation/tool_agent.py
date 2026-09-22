"""Compact model-controlled write -> retrieve -> act loop.

The backend and execution helpers are retained from the research code. This
portable controller does not include historical provider-specific launchers.
"""

import json

from embodied_memorizer import MemoryToolRegistry
from emem_bench.data import resolve_history_image
from . import protocol
from .memory_adapter import MemoryRuntime, _context_object_labels, _merged_label_items
from .planner import extract_json_object, image_to_data_url, post_chat_completion

WRITE_TOOLS = frozenset({"update_object", "remember_event", "remember_scene"})
QUERY_TOOLS = frozenset(
    {"query_spatial", "query_event", "query_scene", "query_experience"}
)


class InvalidPhaseOutput(ValueError):
    """The model returned JSON that violates the active phase contract."""


def validate_calls(payload, phase, schemas):
    if not isinstance(payload, dict) or payload.get("phase") != phase:
        raise InvalidPhaseOutput(f"Expected phase={phase}")
    calls = payload.get("tool_calls")
    if (
        not isinstance(calls, list)
        or len(calls) > 20
        or (phase == "memory" and not calls)
    ):
        raise InvalidPhaseOutput("Invalid tool call list")
    allowed = {(s.get("function") or {}).get("name") for s in schemas}
    for call in calls:
        if not isinstance(call, dict) or call.get("tool") not in allowed:
            raise InvalidPhaseOutput("Tool not available in this phase")
        errors = protocol.validate_memorymodule_tool_call(call, tool_schemas=schemas)
        if errors:
            raise InvalidPhaseOutput("; ".join(errors))


class ToolAgent:
    def __init__(self, args, observation):
        self.args = args
        self.registry = MemoryToolRegistry()
        self.runtime = MemoryRuntime(
            [],
            observation["instruction"],
            current_scene=observation.get("scene"),
            current_namespace=observation.get("episode_id"),
            object_labels=observation.get("object_labels", {}),
            embedding_model=args.embedding_model,
            embedding_device="cpu",
            preload_contexts=False,
            enable_action_filtering=False,
            enable_online_writeback=False,
            enable_progress_action_guard=False,
        )
        self.calls = []

    def schemas(self, names):
        return [
            s
            for s in self.registry.get_schemas()
            if s["function"]["name"] in names
        ]

    def call(self, phase, value, *, image=None, schemas=None, extra_images=()):
        if phase == "context_ingestion":
            instruction = (
                "Read only the current historical observation and feedback. No future "
                "task is available. Store durable, observable locations, states, outcomes "
                "and corrections. Use any subset of the declared write tools, or no "
                "calls if nothing is useful. Do not retrieve or act."
            )
        elif phase == "memory":
            instruction = (
                "Select the memory queries needed for the current task. Do not assume "
                "the answer in a query. You may query raw event history or consolidated "
                "experiences, spatial records or scenes. Do not write or execute robot actions."
            )
        else:
            instruction = (
                "Use the retrieved evidence and current observation to produce one "
                "action_sequence, with 1 to 8 primitive actions, including required "
                "prerequisites. Ground target_label and target_type in observed or "
                "retrieved entities. Do not use memory tools or terminal pseudo-actions."
            )
        if phase == "action":
            contract = protocol.action_phase_response_format(max_actions=8)[
                "json_schema"
            ]["schema"]
        else:
            contract = {
                "phase": phase,
                "reasoning": "brief evidence-based explanation",
                "tool_calls": [
                    {"tool": "<DECLARED_TOOL>", "args": {"<ARGUMENT>": "<VALUE>"}}
                ],
            }
        text = (
            instruction
            + "\nReturn exactly one JSON object matching this contract:\n"
            + json.dumps(contract)
            + "\nDeclared tools:\n"
            + json.dumps(schemas or [])
            + "\nInput:\n"
            + json.dumps(value, ensure_ascii=False)
        )
        content = [{"type": "text", "text": text}]
        images = list(dict.fromkeys([p for p in [image, *extra_images] if p]))
        if not self.args.text_only:
            for path in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
                )
        metadata = {}
        raw = post_chat_completion(
            model_name=self.args.model,
            base_url=self.args.base_url,
            api_key=self.args.api_key,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.args.max_tokens,
            temperature=0,
            timeout=self.args.timeout,
            retries=2,
            retry_delay=2,
            metadata_sink=metadata,
        )
        log = {
            "phase": phase,
            "input": value,
            "image_path": image,
            "retrieved_image_paths": list(extra_images),
            "raw_response": raw,
            "usage": metadata,
        }
        self.calls.append(log)
        try:
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise InvalidPhaseOutput("Expected a JSON object")
            if phase == "action":
                valid, errors = protocol.validate_action_phase_payload(
                    payload, max_actions=8
                )
                if not valid:
                    raise InvalidPhaseOutput("; ".join(errors))
            else:
                payload = protocol.normalize_memorymodule_tool_calls(payload)
                validate_calls(payload, phase, schemas)
        except (ValueError, TypeError, KeyError) as exc:
            log["validation_error"] = str(exc)
            raise InvalidPhaseOutput(str(exc)) from exc
        log["response"] = payload
        return payload

    def ingest(self, contexts, *, episode_path):
        # This stage never receives the future instruction or hidden evaluator labels.
        labels = _context_object_labels(contexts)
        label_items = _merged_label_items({}, labels)
        state = {"seen_objects": set(), "held_object": None}
        for session_index, session in enumerate(contexts):
            for step_index, step in enumerate(protocol.context_steps(session)[:80]):
                observation = protocol.compact_context_observation(
                    session_index=session_index,
                    step_index=step_index,
                    session=session,
                    step=step,
                    object_labels={},
                    context_labels=labels,
                    label_items=label_items,
                    max_visible_objects=40,
                    ingestion_state=state,
                    context_ingestion_mode="aligned",
                )
                observation["scene_scope"] = session.get("scene")
                path = (
                    None
                    if self.args.text_only
                    else resolve_history_image(
                        step.get("image_path"),
                        episode_path=episode_path,
                        data_root=self.args.data_root,
                        session_index=session_index,
                        num_sessions=len(contexts),
                    )
                )
                if not self.args.text_only and not path:
                    raise FileNotFoundError(
                        "Historical RGB is required; use --text-only for a diagnostic run"
                    )
                payload = self.call(
                    "context_ingestion",
                    observation,
                    image=path,
                    schemas=self.schemas(WRITE_TOOLS),
                )
                self.runtime.memory.current_image_path = path
                metadata = protocol.context_step_execution_metadata(
                    memory_runtime=self.runtime,
                    session_index=session_index,
                    step_index=step_index,
                    session=session,
                    step=step,
                    observation=observation,
                )
                self.calls[-1]["tool_results"] = (
                    protocol.execute_memorymodule_tool_results(
                        payload,
                        memory_runtime=self.runtime,
                        registry=self.registry,
                        execution_metadata=metadata,
                    )
                )
                self.runtime.memory.step()
                protocol.advance_context_ingestion_state(
                    observation=observation, ingestion_state=state
                )

    def act(self, env, observation, previous):
        visible = protocol.compact_rollout_observation(
            observation,
            max_context_steps=0,
            max_visible_objects=40,
        )
        memory_payload = self.call(
            "memory",
            {
                "observation": visible,
                "previous_actions": previous,
            },
            image=observation.get("image_path"),
            schemas=self.schemas(QUERY_TOOLS),
        )
        results = protocol.execute_memorymodule_tool_results(
            memory_payload,
            memory_runtime=self.runtime,
            registry=self.registry,
        )
        self.calls[-1]["tool_results"] = results
        retrieved_images = [
            path
            for item in results
            if isinstance(item.get("result"), dict)
            for path in item["result"].get("images", [])
        ]
        action_payload = self.call(
            "action",
            {"observation": visible, "memory_results": results},
            image=observation.get("image_path"),
            extra_images=retrieved_images,
        )
        result = protocol.execute_action_sequence(
            env=env,
            obs=observation,
            payload=action_payload,
            raw_text=json.dumps(action_payload),
            memory_runtime=self.runtime,
            args=self.args,
        )
        next_observation, done, executed, _ = result
        if executed and not done:
            # Model-controlled online writes from visible execution feedback only.
            payload = self.call(
                "context_ingestion",
                {
                    "observation": protocol.compact_rollout_observation(
                        {**next_observation, "instruction": ""},
                        max_context_steps=0,
                        max_visible_objects=40,
                    ),
                    "executed_actions": protocol.model_visible_executed_actions(
                        executed
                    ),
                },
                image=next_observation.get("image_path"),
                schemas=self.schemas(WRITE_TOOLS),
            )
            self.runtime.memory.current_image_path = next_observation.get("image_path")
            self.calls[-1]["tool_results"] = protocol.execute_memorymodule_tool_results(
                payload,
                memory_runtime=self.runtime,
                registry=self.registry,
                execution_metadata={"scene": next_observation.get("scene")},
            )
            self.runtime.memory.step()
        return result
