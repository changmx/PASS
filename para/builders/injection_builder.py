class InjectionBuilder:

    @staticmethod
    def build(model):

        return {
            "Injection": {
                "S (m)": model.s.value,
                "Command":"Injection",
                "Kinetic Energy": model.kinetic_energy.value,
                "Real Particles": model.num_real_particles.value,
                "Emit X": model.emit_x.value,
            }
        }
