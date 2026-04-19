from pathlib import Path

from audio_ecology.config import load_config
from audio_ecology.ingest.inventory import build_and_write_inventory
from audio_ecology.logging_config import configure_logging

configure_logging('INFO')
config = load_config(Path("config_files/config.yaml"))

inventory_df = build_and_write_inventory(config)

print(inventory_df)
