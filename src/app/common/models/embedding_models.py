from langchain.embeddings.base import Embeddings

from typing import (
    Any,
    Dict,
    Literal,
    Union
)

# 需要动态加载这里的包
from langchain_openai import OpenAIEmbeddings
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_xinference import XinferenceEmbeddings
from langchain_community.embeddings import (
    JinaEmbeddings,
    DashScopeEmbeddings, # require pip install dashscope
    ZhipuAIEmbeddings, # require pip install zhipuai
)

def init_embedding_model(
    model : str | None,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    return None























    


