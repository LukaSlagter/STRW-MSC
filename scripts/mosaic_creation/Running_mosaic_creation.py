
import os

os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")
import SF_in_NGC.scripts.photometric_pipeline as PPL

# to_be_done_filters  = ['F356W', 'F150W', 'F212N',          '', '', 'F444W;F405N', 'F430M', 'F182M', 'F444W;F470N',          'F480M', 'F090W', '', '']


def main() -> None:
    PHOTOMETRIC_pipeline = PPL.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )
    PHOTOMETRIC_pipeline.run_pipeline_all(
        filt="F115W",
        overwrite_dic={
            "1/f": True,
            "Jhat calibration": True,
            "Association file": True,
            "Image3Pipeline": True,
        },
    )


if __name__ == "__main__":
    main()


