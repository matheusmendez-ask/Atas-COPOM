"""Unit tests for Observability and OpenTelemetry / Arize Phoenix instrumentation."""

from src.observability.tracer import PipelineTracer, set_span_attributes


class TestTracer:
    """The span helper must work whether or not a collector is configured.

    Every tracer here is built disabled on purpose. An enabled one opens an OTLP
    exporter against PHOENIX_COLLECTOR_ENDPOINT, and a test suite has no business
    writing spans into a real collector. Exporting itself is OpenTelemetry's job,
    not this project's, so there is nothing here worth testing over the wire.
    """

    def test_span_yields_and_accepts_attributes(self):
        tracer = PipelineTracer(project_name="test-project", enabled=False)

        with tracer.span("test_operation", {"doc_count": 5, "stage": "bronze"}) as span:
            set_span_attributes(span, {"custom_attr": "success"})

    def test_disabled_tracer_reports_itself_as_disabled(self):
        tracer = PipelineTracer(enabled=False)

        with tracer.span("disabled_operation", {"foo": "bar"}):
            pass

        assert tracer.enabled is False

    def test_set_span_attributes_tolerates_a_missing_span(self):
        """tracer.span yields None when tracing is off; callers must not have to check."""
        set_span_attributes(None, {"llm.token_count.total": 120})

    def test_set_span_attributes_skips_none_values(self):
        recorded: dict[str, object] = {}

        class FakeSpan:
            def set_attribute(self, key, value):
                recorded[key] = value

        set_span_attributes(FakeSpan(), {"kept": 1, "dropped": None})

        assert recorded == {"kept": 1}
