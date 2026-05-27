from vexrag.usecases.preflight import _vllm_models_list_url


def test_vllm_models_list_url_when_base_includes_v1_suffix() -> None:
    assert (
        _vllm_models_list_url("http://localhost:8017/v1")
        == "http://localhost:8017/v1/models"
    )


def test_vllm_models_list_url_when_base_is_host_root() -> None:
    assert (
        _vllm_models_list_url("http://localhost:8017")
        == "http://localhost:8017/v1/models"
    )
