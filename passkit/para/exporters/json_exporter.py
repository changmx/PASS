import json
from pathlib import Path


class JsonExporter:

    @staticmethod
    def export(data, output_path):

        output_dir = Path(output_path).resolve().parent

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:

            json.dump(data, f, indent=4)
