"""Pytest fixtures for unit and integration testing."""

import os

# Must run before anything imports src: the tracer is a module-level singleton
# built at import time, and with Phoenix enabled the suite exported real spans to
# whatever collector was listening. Test runs polluted the telemetry of a running
# Phoenix with names like "test_operation".
os.environ["ENABLE_PHOENIX"] = "false"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from src.ingestion.schemas import BronzeAtaRecord
from src.vectorstore.embeddings import EmbeddingGenerator
from src.vectorstore.qdrant_manager import QdrantManager


@pytest.fixture
def sample_raw_html() -> str:
    """Provide realistic sample HTML content representing Copom meeting minutes."""
    return """
    <div id="atacompleta">
        <div id="ataconteudo">
            <h3 class="secao">A) Atualização da conjuntura econômica e do cenário do Copom</h3>
            <p class="paragrafo">1. O ambiente externo mostra-se adverso, em função da incerteza sobre a dinâmica da inflação global e os rumos da política monetária nos principais bancos centrais.</p>
            <p class="paragrafo">2. No cenário doméstico, o conjunto dos indicadores de atividade econômica e do mercado de trabalho segue exibindo dinamismo maior do que o esperado.</p>
            <h3 class="secao">B) Cenário prospectivo e balanço de riscos</h3>
            <p class="paragrafo">3. O Copom avalia que a condução da política monetária deve manter-se vigilante e firme para assegurar a convergência da inflação à meta.</p>
            <p class="paragrafo">4. As expectativas de inflação para 2024 e 2025 apuradas pela pesquisa Focus encontram-se em torno de 4,0% e 3,8%, respectivamente.</p>
            <h3 class="secao">C) Decisão de Política Monetária</h3>
            <p class="paragrafo">5. Considerando o balanço de riscos e o cenário básico, o Copom decidiu, por unanimidade, manter a taxa Selic em 10,50% a.a.</p>
        </div>
    </div>
    """


@pytest.fixture
def sample_catalog_response() -> dict:
    """Sample JSON response for catalog endpoint."""
    return {
        "conteudo": [
            {
                "nroReuniao": 280,
                "dataReferencia": "2026-08-05",
                "dataPublicacao": "2026-08-11",
                "titulo": "280ª Reunião - 4-5 agosto, 2026",
            },
            {
                "nroReuniao": 279,
                "dataReferencia": "2026-06-18",
                "dataPublicacao": "2026-06-24",
                "titulo": "279ª Reunião - 17-18 junho, 2026",
            },
        ]
    }


@pytest.fixture
def sample_detail_response(sample_raw_html: str) -> dict:
    """Sample JSON response for detail endpoint."""
    return {
        "conteudo": [
            {
                "nroReuniao": 280,
                "dataReferencia": "2026-08-05",
                "dataPublicacao": "2026-08-11",
                "titulo": "280ª Reunião - 4-5 agosto, 2026",
                "urlPdfAta": "https://www.bcb.gov.br/content/copom/atascopom/Copom280-not20260805280.pdf",
                "textoAta": sample_raw_html,
            }
        ]
    }


@pytest.fixture
def temp_lakehouse_dirs(tmp_path: Path):
    """Provide isolated temporary Bronze and Silver directories."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    return {"bronze": bronze_dir, "silver": silver_dir}


@pytest.fixture
def sample_bronze_record(sample_raw_html: str) -> BronzeAtaRecord:
    """Create a sample valid BronzeAtaRecord."""
    return BronzeAtaRecord.create(
        nro_reuniao=280,
        titulo="280ª Reunião - 4-5 agosto, 2026",
        data_publicacao="2026-08-11",
        raw_content=sample_raw_html,
        source_url="https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes?nro_reuniao=280",
        data_referencia="2026-08-05",
        url_pdf="https://www.bcb.gov.br/content/copom/atascopom/Copom280.pdf",
    )


@pytest.fixture
def in_memory_qdrant() -> QdrantManager:
    """Provide an in-memory QdrantManager instance for lightning-fast testing without Docker."""
    generator = EmbeddingGenerator(provider="fastembed")
    manager = QdrantManager(
        url=":memory:",
        collection_name="test_copom_minutes",
        embedding_generator=generator,
    )
    return manager
