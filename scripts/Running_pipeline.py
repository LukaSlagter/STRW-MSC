import os

os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")
import phot_toolkits as phot_tk

# to_be_done_filters  = ['F356W', 'F150W', 'F212N',          '', '', 'F444W;F405N', 'F430M', 'F182M', 'F444W;F470N',          'F480M', 'F090W', '', '']


def main() -> None:
    JWST_files = phot_tk.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )
    JWST_files.run_pipeline_all(
        filt="F277W",
        overwrite_dic={
            "1/f": True,
            "Jhat calibration": True,
            "Association file": True,
            "Image3Pipeline": True,
        },
        Number_of_dithers=None,
    )


if __name__ == "__main__":
    main()
