from pathlib import Path


class MainBuilder:

    @staticmethod
    def build(model):

        if model.num_proton.value < 0 or model.num_neutron.value < 0:
            raise ValueError("Number of protons and neutrons must be non-negative.")

        if model.num_electron.value == 0:
            raise ValueError("Number of electrons can be positive or negative, but not zero.")

        if model.num_turns.value < 1:
            raise ValueError("Number of turns must be at least 1.")

        if model.circumference.value <= 0:
            raise ValueError("Circumference must be greater than zero.")

        if model.num_gpus.value < 1:
            raise ValueError("Number of GPUs must be at least 1.")

        if not isinstance(model.device_id.value, list) or len(model.device_id.value) != model.num_gpus.value:
            raise ValueError("GPU Device ID must be a list with length equal to the number of GPUs.")

        output_path = Path(model.output_dir.value).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(model.is_plot.value, bool):
            raise ValueError("Generate Plots must be a boolean value.")

        return {
            "Beam Name": model.beam_name.value,
            "Number of Protons": model.num_proton.value,
            "Number of Neutrons": model.num_neutron.value,
            "Number of Electrons": model.num_electron.value,
            "Transition Gamma": model.gamma_t.value,
            "Number of turns": model.num_turns.value,
            "Circumference (m)": model.circumference.value,
            "Number of GPU devices": model.num_gpus.value,
            "Device Id": model.device_id.value,
            "Output directory": str(output_path),
            "Is plot figure": model.is_plot.value,
        }
