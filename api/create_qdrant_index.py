import sys

from loguru import logger

from api.qdrant_api import QdrantMemoryClient
from config.config import Config


class CreateQdrantCollection:
    """初始化 Qdrant collection。"""

    def __init__(self):
        self.client = QdrantMemoryClient(
            collection_name=Config.qdrant.COLLECTION,
            vector_size=Config.qdrant.VECTOR_SIZE,
            distance=Config.qdrant.DISTANCE,
        )

    def create_index(self, index_name: str) -> None:
        """保留旧方法名，内部创建 Qdrant collection。"""
        self.client.collection_name = index_name
        self.client.ensure_collection()
        logger.info(f"Qdrant collection 已就绪: {index_name}")

    def verify_index(self, index_name: str) -> bool:
        """校验 collection 是否存在并输出点数量。"""
        try:
            collection_info = self.client.client.get_collection(index_name)
            points_count = self.client.count()
            logger.info(f"Collection {index_name} verified successfully")
            logger.info(f"Collection status: {collection_info.status}")
            logger.info(f"Total points: {points_count}")
            return True
        except Exception as e:
            logger.error(f"Collection {index_name} does not exist: {e}")
            return False


if __name__ == "__main__":
    try:
        manager = CreateQdrantCollection()
        manager.create_index(Config.qdrant.COLLECTION)
        manager.verify_index(Config.qdrant.COLLECTION)
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)
