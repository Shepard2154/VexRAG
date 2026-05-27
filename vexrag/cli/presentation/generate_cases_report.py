from vexrag.usecases.types import GenerateCasesResult


def format_use_config_snippet(result: GenerateCasesResult) -> str:
    path = result.output_path
    return (
        "  attacks: [ { id: "
        f"{result.attack_id}, params: {{ case_files: ['{path}'], "
        f"adv_per_query: {result.adv_per_query} }} }} ]"
    )


def print_generate_cases_result(result: GenerateCasesResult, *, quiet: bool) -> None:
    if quiet:
        print(result.output_path)
        return

    print(f"Generated {result.display_name} cases")
    print(f"Output: {result.output_path}")
    print(f"Cases: {result.case_count}")
    if result.topic:
        print(f"Topic: {result.topic}")
    print()
    print("Use in config:")
    print(format_use_config_snippet(result))
