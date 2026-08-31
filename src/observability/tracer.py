"""Pipeline OpenTelemetry and Arize Phoenix observability instrumentation."""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from src.config import settings

logger = logging.getLogger(__name__)


class PipelineTracer:
    """Instrumentor for tracing pipeline latency, token usage, and retrieval quality."""

    def __init__(
        self,
        project_name: str | None = None,
        collector_endpoint: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.project_name = project_name or settings.PHOENIX_PROJECT_NAME
        self.collector_endpoint = collector_endpoint or settings.PHOENIX_COLLECTOR_ENDPOINT
        self.enabled = enabled if enabled is not None else settings.ENABLE_PHOENIX
        self.tracer = None

        self._initialize_tracing()

    def _initialize_tracing(self) -> None:
        """Configure OpenTelemetry TracerProvider with Phoenix OTLP HTTP Exporter."""
        if not self.enabled:
            logger.info("Phoenix observability is disabled by configuration.")
            self.tracer = trace.get_tracer("copom-rag-lakehouse-noop")
            return

        resource = Resource.create(
            {
                "service.name": self.project_name,
                # Phoenix routes traces by this exact key. The plain "project.name"
                # used before was ignored without complaint, so every trace landed
                # in Phoenix's "default" project and PHOENIX_PROJECT_NAME did
                # nothing -- harmless with one service, a mess with several.
                "openinference.project.name": self.project_name,
                "environment": "lakehouse-production",
            }
        )

        provider = TracerProvider(resource=resource)

        try:
            otlp_exporter = OTLPSpanExporter(endpoint=self.collector_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(
                f"Arize Phoenix OTLP tracer initialized pointing to {self.collector_endpoint}"
            )
        except Exception as err:
            logger.warning(
                f"Could not connect to Phoenix OTLP endpoint {self.collector_endpoint}: {err}. "
                "Using fallback console/simple span processor."
            )
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("copom-rag-lakehouse")

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[Any, None, None]:
        """Context manager for creating a traceable OpenTelemetry span."""
        if not self.tracer:
            start_time = time.perf_counter()
            yield None
            return

        with self.tracer.start_as_current_span(name) as current_span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        current_span.set_attribute(k, str(v) if isinstance(v, dict | list) else v)
            start_time = time.perf_counter()
            try:
                yield current_span
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                current_span.set_attribute("duration_ms", duration_ms)


def set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """Apply attributes to a span that may be absent when tracing is disabled.

    Attributes computed only after a block runs -- token counts, retrieved
    documents -- cannot be passed to :meth:`PipelineTracer.span`, so they are
    set here instead.
    """
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


# Singleton tracer instance
tracer = PipelineTracer()
