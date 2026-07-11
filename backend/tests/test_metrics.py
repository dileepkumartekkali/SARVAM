from agent_core.llm_adapter import LLMRouter
from agent_core.observability.metrics import llm_latency_seconds, llm_ttfb_seconds, metrics_response

from ._fakes import FakeProvider


def _sample_count(histogram) -> float:
    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                return sample.value
    return 0.0


async def test_complete_with_fallback_records_latency():
    before = _sample_count(llm_latency_seconds)
    router = LLMRouter([FakeProvider("p", chunks=["hello"])])

    await router.complete_with_fallback([{"role": "user", "content": "hi"}])

    assert _sample_count(llm_latency_seconds) == before + 1


async def test_stream_with_fallback_records_ttfb():
    before = _sample_count(llm_ttfb_seconds)
    router = LLMRouter([FakeProvider("p", chunks=["hel", "lo"])])

    async for _ in router.stream_with_fallback([{"role": "user", "content": "hi"}]):
        pass

    assert _sample_count(llm_ttfb_seconds) == before + 1


def test_metrics_response_is_prometheus_text_format():
    body, content_type = metrics_response()
    assert b"maav_llm_latency_seconds" in body
    assert "text/plain" in content_type
