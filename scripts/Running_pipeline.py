import os

os.chdir("/home/slagter/Astronomy_master/STRW-MSC/SF_in_NGC")
import phot_toolkits as phot_tk


def main() -> None:
    JWST_files = phot_tk.Pipeline_level_2_data(
        working_directory="/net/vdesk/data2/slagter/JWST_NIRCAM_#1227_level_2/mastDownload/JWST"
    )
    JWST_files.run_pipeline_all(
        filt="F115W", overwrite_always_run=True, Number_of_systems=None
    )


if __name__ == "__main__":
    main()
