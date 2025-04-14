import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("gpt_api.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")