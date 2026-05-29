class MainBuilder:

    @staticmethod
    def build(model):
        
        return {
            "Name": model.name.value,
            "Number of Protons": model.num_proton.value,
            "Number of Neutrons": model.num_neutron.value,
            "Number of Electrons": model.num_electron.value,
            "Number of Bunches": model.num_bunches.value,
            "Transition Gamma": model.gamma_t.value,
            "Number of turns": model.num_turns.value,
            "Circumference (m)": model.circumference.value,
            "Number of GPUs": model.num_gpus.value,
            "GPU Device ID": model.device_id.value,
            "Output Directory": model.output_dir.value,
            "Generate Plots": model.is_plot.value,
        }
