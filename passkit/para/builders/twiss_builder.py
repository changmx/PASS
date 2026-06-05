from passkit.para.readers.madx_twiss_reader import read_twiss


class TwissBuilder:

    @staticmethod
    def build(model):

        seq, circum = read_twiss(
            twiss_file=model.twiss_file,
            error_file=model.error_file,
            interpolate=model.interpolate,
            num_interp_slice=model.num_interp_slice,
            muz=model.muz,
            dqx=model.dqx,
            dqy=model.dqy,
        )

        return seq, circum
