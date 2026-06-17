from __future__ import annotations

import json
from pathlib import Path

from soccer_ev_model.goal_model_data import write_audit_report


def main() -> None:
    output = Path("reports/goal_model_data_audit.json")
    report = write_audit_report(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Wrote", output)


if __name__ == "__main__":
    main()
