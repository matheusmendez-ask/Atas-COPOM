"""Unit tests for Observability and OpenTelemetry / Arize Phoenix instrumentation."""

from src.observability.tracer import PipelineTracer


class TestTracer:
    """Test suite for PipelineTracer instrumentation."""

    def test_tracer_span_context_manager(self):
        """Verify that tracer.span creates spans and adds attributes without errors."""
        tracer_inst = PipelineTracer(project_name="test-project", enabled=True)
        with tracer_inst.span("test_operation", {"doc_count": 5, "stage": "bronze"}) as span:
            # Simulate work
            val = 1 + 1
            assert val == 2
            if span:
                span.set_attribute("custom_attr", "success")

    def test_tracer_disabled_mode(self):
        """Verify tracer gracefully falls back when disabled."""
        disabled_tracer = PipelineTracer(enabled=False)
        with disabled_tracer.span("disabled_operation", {"foo": "bar"}):
            pass
        assert disabled_tracer.enabled is False
