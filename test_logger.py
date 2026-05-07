from common import logger

def test_logger():
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    print("\nLogger 类型:", type(logger))
    print("Logger 可用方法:", [attr for attr in dir(logger) if not attr.startswith('_')])

if __name__ == "__main__":
    test_logger()
