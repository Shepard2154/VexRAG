from .nq_loading import BenchmarkExample


def benchmark_example_from_retrieved_context(context: str) -> BenchmarkExample | None:
    text = context.strip()
    if not text:
        return None
    return BenchmarkExample(
        sample_id=-1,
        question="",
        answer="",
        context=text,
    )
