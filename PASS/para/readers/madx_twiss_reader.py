from passkit.para.get_twiss_from_madx import (
    get_twiss_from_madx_twissfile,
    get_twiss_interpolate_from_madx_twissfile,
)


def read_twiss(
    twiss_file,
    error_file="",
    interpolate=False,
    num_interp_slice=100,
    muz=0.001,
    dqx=0.0,
    dqy=0.0,
):

    if interpolate:

        return get_twiss_interpolate_from_madx_twissfile(
            twiss_file_path=twiss_file,
            error_file_path=error_file,
            num_interp_slice=num_interp_slice,
            muz=muz,
            DQx=dqx,
            DQy=dqy,
        )

    else:

        return get_twiss_from_madx_twissfile(
            twiss_file_path=twiss_file,
            error_file_path=error_file,
            muz=muz,
            DQx=dqx,
            DQy=dqy,
        )