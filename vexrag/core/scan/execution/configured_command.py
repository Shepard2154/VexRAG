from collections.abc import Callable
from typing import Any, cast

from vexrag.core.scan.contracts import ScanCaseReport, ScanReport


class ConfiguredScanCommand:
    """Wires a runner with requests and scan config."""

    def __init__(self, runner: Any, requests: Any, scan_config: Any) -> None:
        self.runner = runner
        self.requests = requests
        self.scan_config = scan_config

    def run(
        self,
        *,
        on_case_complete: Callable[[ScanCaseReport], None] | None = None,
    ) -> ScanReport:
        return cast(
            ScanReport,
            self.runner.run(
                self.requests,
                self.scan_config,
                on_case_complete=on_case_complete,
            ),
        )
