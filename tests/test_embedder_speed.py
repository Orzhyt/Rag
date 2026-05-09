"""本地 Embedding 模型速度基准测试（pytorch / onnx 双后端对比）"""

import os
import time
import statistics

import pytest

os.environ.setdefault("TORCHVISION_DISABLE_EXTENSION", "1")

from dotenv import load_dotenv
from milvus import EmbeddingModel

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL") or None
DEVICE = os.getenv("EMBEDDING_DEVICE") or "auto"


def _format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


@pytest.fixture(scope="session")
def embedder(request):
    return EmbeddingModel(model_name=EMBEDDING_MODEL_NAME, device=DEVICE)


def test_single_encode_speed(embedder):
    """测试单条文本编码耗时"""
    text = "这是一个用于测试本地embedding模型编码速度的示例文本。"
    times = []
    for _ in range(20):
        start = time.perf_counter()
        embedder.encode([text])
        times.append(time.perf_counter() - start)

    avg = statistics.mean(times)
    med = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"\n[单条编码 x20] avg={_format_time(avg)} med={_format_time(med)} p95={_format_time(p95)}")
    assert avg < 5, "单条编码平均耗时超过 5s"


def test_batch_encode_speed(embedder):
    """测试不同批量大小下的编码吞吐量"""
    sample = """  检索增强生成技术（Retrieval-Augmented                                                                                                                                                                                             
  Generation，简称RAG）是近年来自然语言处理领域最具影响力的技术范式之一。它的核心思想是将大规模语言模型的生成能力与外部知识库的检索能力相结合，从而在不重新训练模型的前提下，为模型注入领域特定知识和实时信息。传统的大语言模型虽然
  具备强大的语言理解和生成能力，但其知识来源于训练数据，存在知识截止日期的限制，且无法获取最新的事实信息。此外，模型在处理专业领域问题时，往往因为训练数据中相关信息的匮乏而产生幻觉，即生成看似合理但实际上不正确的内容。          
                                                            
  RAG系统通常由三个核心组件构成：检索器、重排器和生成器。检索器负责根据用户查询从知识库中召回相关文档片段，通常采用基于向量的语义检索或关键词匹配的稀疏检索，以及两者的混合检索策略。重排器对检索结果进行精细排序，利用交叉编码器等
  模型计算查询与候选文档的深度相关性分数，从而过滤掉语义相似但实际不相关的噪声文档。生成器则将重排后的文档作为上下文，与用户查询一起输入语言模型，生成最终的回答。

  在实际工程落地中，RAG系统面临诸多挑战。首先是文档切分策略的选择：过长的切片会引入噪声并增加推理延迟，过短的切片则会破坏语义完整性，导致关键信息被截断。其次是向量模型的选择与优化：不同规模的嵌入模型在语义表征能力、推理速度和资
  源占用之间存在显著权衡，需要根据业务场景的延迟要求和准确度需求进行取舍。此外，知识库的增量更新、多模态文档的处理、检索结果与生成内容的一致性校验，以及长上下文下的信息压缩与筛选，都是生产环境中必须解决的关键问题。

  随着大模型上下文窗口的不断扩展，一种观点认为RAG将被长上下文模型取代。然而，长上下文并不意味着长推理——模型在处理超长输入时，对中间位置信息的注意力衰减现象已被多项研究证实，即所谓的"迷失在中间"效应。因此，即使上下文窗口足以容纳
  所有相关文档，精准的检索与排序仍然是确保模型关注正确信息、避免被无关内容干扰的必要手段。RAG与长上下文并非对立关系，而是互补关系：检索负责精准定位，上下文负责充分理解，两者结合才能实现既准又深的知识增强效果。

  大约 750 个中文字符，对应 Qwen 分词器下约 500 token。"""
    batch_sizes = [1, 4, 8]

    print(f"\n{'batch_size':>10} {'总耗时':>10} {'条/秒':>10} {'ms/条':>10}")
    print("-" * 45)

    for bs in batch_sizes:
        texts = [sample] * bs
        start = time.perf_counter()
        embedder.encode(texts)
        elapsed = time.perf_counter() - start
        throughput = bs / elapsed
        ms_per_item = elapsed / bs * 1000
        print(f"{bs:>10} {_format_time(elapsed):>10} {throughput:>10.1f} {ms_per_item:>10.1f}")
