import logging
import sys

def setup_logger(name):
    """
    Set up a unified logging configuration.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure the logging system
    logger = logging.getLogger(name)
    
    # Return early if the logger has already been configured
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    
    # Set the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Attach the handler to the logger
    logger.addHandler(console_handler)

    return logger
