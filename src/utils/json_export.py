import json
from typing import Dict, Any


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
