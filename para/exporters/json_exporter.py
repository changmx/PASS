import json


class JsonExporter:

    @staticmethod
    def export(data, path):

        with open(path, "w") as f:

            json.dump(data, f, indent=4)