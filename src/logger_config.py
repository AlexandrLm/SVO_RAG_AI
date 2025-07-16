import logging
import sys

def setup_logging():
    """
    Настраивает и конфигурирует стандартный логгер Python.
    """
    # Создаем форматтер, который будет добавлять время, уровень и сообщение
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Создаем обработчик, который будет выводить логи в консоль (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Получаем корневой логгер, устанавливаем ему уровень INFO
    # и добавляем наш обработчик
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Очищаем существующих обработчиков, чтобы избежать дублирования
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)

    print("Логгирование успешно настроено.") 