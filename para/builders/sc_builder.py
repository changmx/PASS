class SpaceChargeBuilder:

    @staticmethod
    def build(model):

        if not model.enable:

            return {}

        return {
            "Space-charge simulation parameters": {
                "Is enable space charge": model.enable,
                "Number of slices": model.num_slices,
                "Slice model": model.slice_model,
                "Field solver": model.field_solver,
            }
        }
