import ezpadova


def Find_paths_Stellar_Tracks(N_age, logAge, N_Z, Z):
    return f"Stellar_tracks/track_data/ISOCHRONES_logAge_{N_age}_{logAge[0]}_{logAge[1]}_Z_{N_Z}_{Z[0]}_{Z[1]}"


def Download_Stellar_Tracks_Parsec12S(N_age, logAge, N_Z, Z):
    # The following are the standard ones on the website DO NOT CHANGE!
    track_version = "parsec_CAF09_v1.2S"  #
    track_colibri = "parsec_CAF09_v1.2S_S_LMC_08_web"
    extinction_av = 0.0
    imf_file = "tab_imf/imf_kroupa_orig.dat"

    photsys_files_jwst_narrow = "tab_mag_odfnew/tab_mag_jwst_narrow.dat"
    photsys_files_jwst_widemedium = (
        "tab_mag_odfnew/tab_mag_jwst_nircam_widemedium_dec23.dat"
    )

    r_widemedium = ezpadova.get_isochrones(
        logage=(logAge[0], logAge[1], N_age),
        Z=(Z[0], Z[1], N_Z),
        photsys_file=photsys_files_jwst_widemedium,
        track_version=track_version,
        track_colibri=track_colibri,
        extinction_av=extinction_av,
        imf_file=imf_file,
    )

    r_narrow = ezpadova.get_isochrones(
        logage=(logAge[0], logAge[1], N_age),
        Z=(Z[0], Z[1], N_Z),
        photsys_file=photsys_files_jwst_narrow,
        track_version=track_version,
        track_colibri=track_colibri,
        extinction_av=extinction_av,
        imf_file=imf_file,
    )

    want_to_keep_filters_narrow = [
        "F164Nmag",
        "F187Nmag",
        "F212Nmag",
        "F323Nmag",
        "F405Nmag",
        "F466Nmag",
        "F470Nmag",
    ]

    Combined_data = r_widemedium
    Combined_data[want_to_keep_filters_narrow] = r_narrow[want_to_keep_filters_narrow]
    savename = Find_paths_Stellar_Tracks(N_age, logAge, N_Z, Z)

    Combined_data.to_csv(
        savename,
        sep=" ",
        index=False,
    )
    print("---FINISHED---")
    return savename
