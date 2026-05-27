from vexrag.usecases.types import DoctorResult


def print_doctor_result(result: DoctorResult) -> None:
    print("VexRAG Doctor")
    print()
    for check in result.checks:
        icon = "OK" if check.ok else "FAIL"
        print(f"[{icon}] {check.name}")
        if check.error:
            print(f"  -> {check.error}")

    print()
    if result.passed:
        print("Doctor verdict: PASS")
        return
    print(f"Doctor verdict: FAIL ({result.failed_count} check(s) failed)")
