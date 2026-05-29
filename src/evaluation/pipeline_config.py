"""
Pipeline configuration dataclass and all ablation variants for the MiNIonek RAG study.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    name: str
    use_query_rewrite: bool = True
    retrieval_mode: str = "hybrid"  # "hybrid" | "dense_only"
    use_rerank: bool = True
    rerank_steps: int = 1  # 1=single, 2=two-step (60->30->15)
    content_type: str = "facts"  # "facts" | "chunks"
    n_retrieve: int = 30  # candidates after retrieval
    n_final: int = 15  # after reranking
    two_stage: bool = False  # doc-level first, then fact/chunk level
    two_stage_content: str = "facts"  # "facts" | "chunks" for stage 2
    description: str = ""


ALL_VARIANTS: dict[str, PipelineConfig] = {
    "baseline": PipelineConfig(
        name="baseline",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=1,
        content_type="facts",
        n_retrieve=30,
        n_final=15,
        two_stage=False,
        description="Hybrid retrieval + query rewrite + single cross-encoder rerank, facts collection.",
    ),
    "v1_vector_only": PipelineConfig(
        name="v1_vector_only",
        use_query_rewrite=True,
        retrieval_mode="dense_only",
        use_rerank=True,
        rerank_steps=1,
        content_type="facts",
        n_retrieve=30,
        n_final=15,
        two_stage=False,
        description="Dense-only retrieval (no BM25/sparse) + query rewrite + rerank.",
    ),
    "v2_no_rerank": PipelineConfig(
        name="v2_no_rerank",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=False,
        rerank_steps=1,
        content_type="facts",
        n_retrieve=15,
        n_final=15,
        two_stage=False,
        description="Hybrid retrieval + query rewrite, NO reranking — raw RRF top-15.",
    ),
    "v3_two_step_rerank": PipelineConfig(
        name="v3_two_step_rerank",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=2,
        content_type="facts",
        n_retrieve=60,
        n_final=15,
        two_stage=False,
        description="Hybrid retrieval 60->30->15 two-step cross-encoder rerank.",
    ),
    "v4_chunks": PipelineConfig(
        name="v4_chunks",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=1,
        content_type="chunks",
        n_retrieve=30,
        n_final=15,
        two_stage=False,
        description="Hybrid retrieval + rerank over raw text chunks (mini_chunks collection).",
    ),
    "v5_no_rewrite": PipelineConfig(
        name="v5_no_rewrite",
        use_query_rewrite=False,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=1,
        content_type="facts",
        n_retrieve=30,
        n_final=15,
        two_stage=False,
        description="Hybrid retrieval + rerank, NO query rewriting — raw user query.",
    ),
    "v6_two_stage_facts": PipelineConfig(
        name="v6_two_stage_facts",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=1,
        content_type="facts",
        n_retrieve=30,
        n_final=15,
        two_stage=True,
        two_stage_content="facts",
        description="Two-stage: URL-level stage 1, then fact-level retrieval+rerank within top-10 URLs.",
    ),
    "v7_two_stage_chunks": PipelineConfig(
        name="v7_two_stage_chunks",
        use_query_rewrite=True,
        retrieval_mode="hybrid",
        use_rerank=True,
        rerank_steps=1,
        content_type="chunks",
        n_retrieve=30,
        n_final=15,
        two_stage=True,
        two_stage_content="chunks",
        description="Two-stage: URL-level stage 1, then chunk-level retrieval+rerank within top-10 URLs.",
    ),
}
