class InjectionBuilder:

    @staticmethod
    def build(model):
        """
        Build the injection JSON node.

        Output structure:
        {
            "Injection": {
                "S (m)": 0.0,
                "Command": "Injection",
                "bunch0": { ... },
                "bunch1": { ... },
                ...
            }
        }
        """

        injection_dict = {
            "S (m)": model.s.value,
            "Command": model.command.value,
        }

        for i, bunch in enumerate(model.bunches):
            bunch_key = f"bunch{i}"
            injection_dict[bunch_key] = {
                "Kinetic Energy per Nucleon (eV/u)": bunch.kinetic_energy.value,
                "Number of Real Particles": bunch.num_real_particles.value,
                "Number of Macro Particles": bunch.num_macro_articles.value,
                "Is Load Distribution from File": bunch.is_load_from_file.value,
                "Distribution File Path": bunch.file_path.value,
                "Total Injection Turns": bunch.injection_turn.value,
                "Injection Interval": bunch.injection_interval.value,
                "Alpha x": bunch.alpha_x.value,
                "Alpha y": bunch.alpha_y.value,
                "Beta x (m)": bunch.beta_x.value,
                "Beta y (m)": bunch.beta_y.value,
                "Emittance x (m'rad)": bunch.emit_x.value,
                "Emittance y (m'rad)": bunch.emit_y.value,
                "Dx (m)": bunch.Dpx.value,
                "Dpx": bunch.Dpx.value,
                "Sigma z (m)": bunch.sigma_z.value,
                "Sigma dp/p": bunch.dp.value,
                "Transverse dist": bunch.dist_trans.value,
                "Longitudinal dist": bunch.dist_longi.value,
                "RF Voltage (V)": bunch.rf_voltage.value,
                "RF Phase (rad)": bunch.rf_phase.value,
                "Harmonic Number": bunch.num_harmonics.value,
                "Harmonic ID of this bunch": bunch.harmonic_id.value,
                "RF S Position Refer to Inj. Point (m)": bunch.rf_delta_dist.value,
                "Offset x": {
                    "Is Offset": bunch.is_offset_x.value,
                    "Is Load From File": bunch.is_load_offset_x_from_file.value,
                    "File Path": bunch.offset_x_filepath.value,
                    "File Time Kind": bunch.offset_x_file_timekind.value,
                    "Offset Position (m)": bunch.offset_x_position.value,
                    "Offset Momentum (rad)": bunch.offset_x_momentum.value,
                },
                "Offset y": {
                    "Is Offset": bunch.is_offset_y.value,
                    "Is Load From File": bunch.is_load_offset_y_from_file.value,
                    "File Path": bunch.offset_y_filepath.value,
                    "File Time Kind": bunch.offset_y_file_timekind.value,
                    "Offset Position (m)": bunch.offset_y_position.value,
                    "Offset Momentum (rad)": bunch.offset_y_momentum.value,
                },
                "Is Save Initial Distribution": bunch.save_init_dist.value,
                "Insert Particle Coordinate": bunch.insert_particle.value,
            }

        return {"Injection": injection_dict}
