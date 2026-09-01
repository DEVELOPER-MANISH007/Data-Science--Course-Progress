import logging

## logging settings

logging.basicConfig(
     
    level=logging.DEBUG,
    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("app1.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Arthimatic App")

def add(a,b):
    result = a+b
    logger.debug(f"adding,{a} + {b} = {result} ")
    return  result

def subtract(a,b):
    result = a-b
    logger.debug(f"adding,{a} - {b} = {result} ")
    return  result

def MUL(a,b):
    result = a*b
    logger.debug(f"adding,{a} * {b} = {result} ")
    return  result

def div(a,b):
    try:
        result = a/b
        logger.debug(f"adding,{a} / {b} = {result} ")
        return  result
    except ZeroDivisionError:
        logger.error("Division by zero error")

add(23,32)
subtract(32,21)
MUL(23,2)
div(45,5)
