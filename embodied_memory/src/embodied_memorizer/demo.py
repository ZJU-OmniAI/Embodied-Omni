"""Small deterministic memory example; no dataset, weights, API or GPU needed."""

import json
from .memory import MemorySystem


def main():
    memory = MemorySystem(
        {"embedding_model": "lexical-hash", "embedding_device": "cpu"}
    )
    memory.update_spatial(
        "Mug", relations=[{"target": "CounterTop", "type": "on"}], scope="kitchen"
    )
    memory.step()
    memory.update_spatial(
        "Mug", relations=[{"target": "Cabinet", "type": "in"}], scope="kitchen"
    )
    print(json.dumps(memory.query_spatial("Mug", top_k=3, scope="kitchen"), indent=2))


if __name__ == "__main__":
    main()
