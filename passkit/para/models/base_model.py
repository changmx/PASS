from dataclasses import asdict

class BaseModel:

    def to_dict(self):

        return asdict(self)