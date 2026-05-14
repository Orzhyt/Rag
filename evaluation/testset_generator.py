"""基于 ragas TestsetGenerator 的测试集自动生成模块。

从文档自动生成问答对，替代手工编写 JSONL 测试数据集。

用法:
    # 从文档目录生成
    from evaluation.testset_generator import load_documents_from_dir, generate_testset
    docs = load_documents_from_dir("./data/legal")
    testset = generate_testset(docs, testset_size=10)

    # 从 Milvus 已有切片生成
    from evaluation.testset_generator import load_chunks_from_milvus, generate_testset_from_chunks
    chunks = load_chunks_from_milvus(milvus_service, collection_names=["rag_collection"])
    testset = generate_testset_from_chunks(chunks, testset_size=10)
"""

import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document as LCDocument
from ragas.testset.synthesizers.generate import TestsetGenerator

from common.logger import setup_logger

logger = setup_logger("evaluation.testset_generator")


def patch_chinese_prompts():
    """将 ragas TestsetGenerator 内部的英文提示词替换为中文。

    通过 monkey-patch 将 instruction 和 examples 替换为中文版本，
    使 LLM 生成中文问答对。
    """
    from ragas.testset.persona import Persona, PersonaGenerationPrompt
    from ragas.prompt import StringIO
    from ragas.testset.synthesizers.prompts import (
        ThemesPersonasInput,
        ThemesPersonasMatchingPrompt,
    )
    from ragas.testset.synthesizers.single_hop.prompts import (
        QueryAnswerGenerationPrompt as SingleHopQA,
        QueryCondition as SingleHopCondition,
        GeneratedQueryAnswer as SingleHopQAOutput,
    )
    from ragas.testset.synthesizers.multi_hop.prompts import (
        ConceptCombinationPrompt,
        ConceptsList,
        QueryAnswerGenerationPrompt as MultiHopQA,
        QueryConditions as MultiHopCondition,
        GeneratedQueryAnswer as MultiHopQAOutput,
    )
    from ragas.testset.transforms.extractors.llm_based import (
        SummaryExtractorPrompt,
        KeyphrasesExtractorPrompt,
        TopicDescriptionPrompt,
        ThemesAndConceptsExtractorPrompt,
    )

    # --- Persona 生成 ---
    PersonaGenerationPrompt.instruction = (
        "根据提供的文档摘要，生成一个可能会与该内容互动或从中受益的用户画像。"
        "包含一个独特的名称和简洁的角色描述。"
        "请用中文生成。"
    )
    PersonaGenerationPrompt.examples = [
        (
            StringIO(text="数字营销指南介绍了在各种在线平台上吸引受众的策略。"),
            Persona(name="数字营销专员", role_description="专注于在线吸引受众和品牌增长。"),
        )
    ]

    # --- Persona-Theme 匹配 ---
    ThemesPersonasMatchingPrompt.instruction = (
        "给定一组主题和用户画像及其角色描述，根据角色描述将每个画像与相关主题关联。"
        "请用中文处理。"
    )

    # --- 单跳问答生成 ---
    SingleHopQA.instruction = (
        "根据指定条件（用户画像、关键短语、问题风格、问题长度）和提供的上下文，"
        "生成一个单跳问题和答案。答案必须完全忠实于上下文，仅使用上下文中直接包含的信息。\n"
        "### 指令：\n"
        "1. **生成问题**：基于上下文、用户画像、关键短语、风格和长度，创建一个符合用户画像视角并包含关键短语的问题。\n"
        "2. **生成答案**：仅使用提供的上下文内容，构建对问题的详细回答。不要添加上下文中未包含或无法推断的信息。\n"
        "3. **额外上下文**（如提供）：如果 llm_context 已提供，将其作为生成何种类型问题的指导"
        "（例如对比问题、操作方法问题、应用类问题），以及如何组织答案。仍需确保内容仅来自提供的上下文。\n"
        "请用中文生成问题和答案。"
    )
    SingleHopQA.examples = [
        (
            SingleHopCondition(
                persona=Persona(name="安全监管员", role_description="关注企业安全生产合规和风险管控。"),
                term="重大事故隐患",
                query_style="正式",
                query_length="中等",
                context="重大事故隐患是指在生产作业中存在的可能导致重大人身伤亡或重大经济损失的事故隐患。"
                "企业应当建立重大事故隐患排查治理制度。",
            ),
            SingleHopQAOutput(
                query="什么是重大事故隐患？企业应当建立什么制度？",
                answer="重大事故隐患是指在生产作业中存在的可能导致重大人身伤亡或重大经济损失的事故隐患。企业应当建立重大事故隐患排查治理制度。",
            ),
        ),
    ]

    # --- 多跳概念组合 ---
    ConceptCombinationPrompt.instruction = (
        "通过将来自至少两个不同列表的概念进行配对，形成组合。\n"
        "**指令：**\n"
        "- 审查每个节点的概念。\n"
        "- 识别可以逻辑上关联或对比的概念。\n"
        "- 形成涉及来自不同节点概念的组合。\n"
        "- 每个组合应包含至少两个或更多节点中的各一个概念。\n"
        "- 清晰简洁地列出组合。\n"
        "- 不要重复相同的组合。"
    )

    # --- 多跳问答生成 ---
    MultiHopQA.instruction = (
        "根据指定条件（用户画像、主题、问题风格、问题长度）和提供的上下文，"
        "生成一个多跳问题和答案。主题代表从上下文中提取或生成的短语集合，"
        "突出所选上下文适合多跳问题创建的特性。确保问题显式地包含这些主题。\n"
        "### 指令：\n"
        "1. **生成多跳问题**：使用提供的上下文片段和主题，形成一个需要结合多个片段信息的问题"
        "（例如 `<1-hop>` 和 `<2-hop>`）。确保问题显式地包含一个或多个主题，并反映它们与上下文的相关性。\n"
        "2. **生成答案**：仅使用提供的上下文内容，创建对问题的详细且忠实的回答。避免添加非直接存在或无法推断的信息。\n"
        "3. **多跳上下文标签**：\n"
        "   - 每个上下文片段标记为 `<1-hop>`、`<2-hop>` 等。\n"
        "   - 确保问题使用至少两个片段的信息，并将它们有意义地连接。\n"
        "4. **额外上下文**（如提供）：如果 llm_context 已提供，将其作为生成何种类型问题的指导"
        "（例如对比问题、因果关系问题、应用类问题），以及如何组织答案。仍需确保内容仅来自提供的上下文。\n"
        "请用中文生成问题和答案。"
    )
    MultiHopQA.examples = [
        (
            MultiHopCondition(
                persona=Persona(name="安全监管员", role_description="关注企业安全生产合规和风险管控。"),
                themes=["重大事故隐患", "安全风险管理"],
                query_style="正式",
                query_length="中等",
                context=[
                    "<1-hop> 生产经营单位应当建立安全风险分级管控制度，采取相应管控措施。",
                    '<2-hop> 涉及“两重点一重大”的生产装置外部安全防护距离不符合国家标准要求，应当判定为重大事故隐患。',
                ],
            ),
            MultiHopQAOutput(
                query="安全风险分级管控与重大事故隐患判定之间有什么关系？",
                answer='生产经营单位应当建立安全风险分级管控制度并采取管控措施。当涉及"两重点一重大"的生产装置外部安全防护距离不符合国家标准要求时，应当判定为重大事故隐患，这体现了风险管控不到位可能导致重大隐患的关联。',
            ),
        ),
    ]

    # --- Transforms 提取器 ---
    SummaryExtractorPrompt.instruction = "用中文将给定文本总结为不超过10句话的摘要。"
    KeyphrasesExtractorPrompt.instruction = "从给定文本中提取最多 max_num 个中文关键短语。"
    TopicDescriptionPrompt.instruction = "用中文简要描述以下文本讨论的主要主题。"
    ThemesAndConceptsExtractorPrompt.instruction = "从给定文本中提取主要的中文主题和概念。"

    logger.info("已将 ragas 提示词替换为中文版本")


def load_documents_from_dir(data_dir: str) -> List[LCDocument]:
    """扫描文档目录，解析并转为 LangChain Document 列表。

    使用项目现有的 FileScanner + DocumentParser，每个文件解析后的
    完整文本作为一个 Document（不分块，分块由 ragas TestsetGenerator 内部完成）。
    """
    from data_pipeline.scanner import FileScanner
    from data_pipeline.parser import create_default_registry

    scanner = FileScanner()
    file_paths = scanner.scan_directory(data_dir)
    logger.info("扫描到 %d 个文件", len(file_paths))

    registry = create_default_registry()
    documents: List[LCDocument] = []

    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        parser = registry.get(ext)
        if parser is None:
            logger.warning("无解析器: %s [%s]", ext, file_path)
            continue

        try:
            content, extra_metadata = parser.parse_with_metadata(file_path)
        except Exception as e:
            logger.error("解析失败 [%s]: %s", file_path, e)
            continue

        if not content or not content.strip():
            continue

        metadata = {"source": file_path, "file_name": Path(file_path).name}
        metadata.update(extra_metadata or {})
        documents.append(LCDocument(page_content=content, metadata=metadata))

    logger.info("成功解析 %d 个文档", len(documents))
    return documents


def load_chunks_from_milvus(
    milvus_service,
    collection_names: Optional[List[str]] = None,
) -> List[LCDocument]:
    """从 Milvus 中查询已有切片，转为 LangChain Document 列表。

    使用 MilvusService.query() 获取所有记录，提取 content 和 metadata。
    """
    from pymilvus import MilvusClient as PyMilvusClient

    collections = collection_names or [milvus_service.client.collection_name]

    documents: List[LCDocument] = []
    for coll_name in collections:
        try:
            results = milvus_service.client.query(
                collection_name=coll_name,
                filter="",
                output_fields=["content", "source_file", "file_name", "file_type", "chunk_index"],
            )
        except Exception as e:
            logger.error("查询 Milvus collection [%s] 失败: %s", coll_name, e)
            continue

        for row in results:
            content = row.get("content", "")
            if not content or not content.strip():
                continue
            metadata = {
                "source": row.get("source_file", ""),
                "file_name": row.get("file_name", ""),
                "file_type": row.get("file_type", ""),
                "chunk_index": row.get("chunk_index", 0),
                "collection": coll_name,
            }
            documents.append(LCDocument(page_content=content, metadata=metadata))

    logger.info("从 Milvus 加载 %d 个切片", len(documents))
    return documents


def _build_query_distribution(ragas_llm, llm_context=None):
    """构建 5:3:2 比例的 query_distribution。

    SingleHopSpecific : MultiHopAbstract : MultiHopSpecific = 5 : 3 : 2
    """
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

    return [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, llm_context=llm_context), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm, llm_context=llm_context), 0.3),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm, llm_context=llm_context), 0.2),
    ]


def generate_testset(
    documents: List[LCDocument],
    testset_size: int,
    ragas_llm,
    ragas_embeddings,
    llm_context: Optional[str] = None,
) -> "Testset":
    """从原始文档生成测试集。

    ragas 会自动完成：分块 → 知识图谱构建 → 问答对生成。
    """
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        llm_context=llm_context,
    )

    patch_chinese_prompts()
    query_distribution = _build_query_distribution(ragas_llm, llm_context)

    logger.info("开始生成测试集（%d 个文档 → %d 条样本，比例 5:3:2）...", len(documents), testset_size)
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=testset_size,
        query_distribution=query_distribution,
    )
    logger.info("测试集生成完成，共 %d 条", len(testset.samples))
    return testset


def generate_testset_from_chunks(
    chunks: List[LCDocument],
    testset_size: int,
    ragas_llm,
    ragas_embeddings,
    llm_context: Optional[str] = None,
) -> "Testset":
    """从已有切片生成测试集。

    跳过 ragas 内部分块，直接基于 chunks 构建知识图谱并生成问答对。
    """
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        llm_context=llm_context,
    )

    patch_chinese_prompts()
    query_distribution = _build_query_distribution(ragas_llm, llm_context)

    logger.info("开始从切片生成测试集（%d 个切片 → %d 条样本，比例 5:3:2）...", len(chunks), testset_size)
    testset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=testset_size,
        query_distribution=query_distribution,
    )
    logger.info("测试集生成完成，共 %d 条", len(testset.samples))
    return testset


def save_testset_to_jsonl(testset: "Testset", output_path: str) -> str:
    """将 Testset 保存为 JSONL 文件，兼容现有 load_dataset() 格式。

    每行格式: {"user_input": "...", "reference": "...", "reference_contexts": [...]}
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in testset.samples:
        eval_sample = sample.eval_sample
        record = {
            "user_input": eval_sample.user_input,
            "reference": eval_sample.reference,
            "reference_contexts": eval_sample.reference_contexts or [],
        }
        # 附加元数据方便溯源
        if eval_sample.persona_name:
            record["persona_name"] = eval_sample.persona_name
        if eval_sample.query_style:
            record["query_style"] = eval_sample.query_style
        if eval_sample.query_length:
            record["query_length"] = eval_sample.query_length
        record["synthesizer_name"] = sample.synthesizer_name
        records.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("测试集已保存到 %s（%d 条）", output_path, len(records))
    return str(out_path)
