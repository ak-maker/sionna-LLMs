import os
from getpass import getpass
from pathlib import Path
from typing import Any
from click import UsageError


CONFIG_PATH = Path("<Your Path of .env file in RAG>")

# TODO: Refactor ENV variables with SGPT_ prefix.
DEFAULT_CONFIG = {
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.getenv("OPENAI_API_KEY", None)
    # New features might add their own config variables here.
}


class Config(dict):  # type: ignore
    def __init__(self, config_path: Path, **defaults: Any):
        self.config_path = config_path

        if self._exists:
            self._read()
            has_new_config = False
            for key, value in defaults.items():
                if key not in self:
                    has_new_config = True
                    self[key] = value
            if has_new_config:
                self._write()
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't write API key to config file if it is in the environment.
            if not defaults.get("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
                __api_key = getpass(prompt="Please enter your OpenAI API key: ")
                defaults["OPENAI_API_KEY"] = __api_key
            super().__init__(**defaults)
            self._write()

    @property
    def _exists(self) -> bool:
        return self.config_path.exists()

    def _write(self) -> None:
        with open(self.config_path, "w", encoding="utf-8") as file:
            string_config = ""
            for key, value in self.items():
                string_config += f"{key}={value}\n"
            file.write(string_config)

    def _read(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=")
                    self[key] = value

    def get(self, key: str, default=None) -> str:  # type: ignore
        # Prioritize config file over environment variables.
        value = super().get(key) or os.getenv(key)
        to_return = value if value is not None else default
        if to_return is None:
            raise UsageError(f"Missing config key: {key}")
        return to_return


cfg = Config(CONFIG_PATH, **DEFAULT_CONFIG)
