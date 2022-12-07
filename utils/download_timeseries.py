from os import makedirs, listdir
from pathlib import Path
from os.path import basename, join, exists
from datetime import datetime
import pytz
import awswrangler as wr
from typing import Union


def download_timeseries_data(
    bucket_dir: Union[str, Path], output_dir: Union[str, Path] = "input/timeseries_data", date_begin: str = "01.10.22 00:00"
) -> None:
    """Downloads all timeseries data from 'bucket_dir' since 'date_begin' and saves them into 'output_dir'.
    AWS access needs to be established first. For example over AWS CLI Configure.

    Args:
        bucket_dir (str): Directory to AWS s3 bucket. e.g. "s3://example/dir/subdir/"
        output_dir (str, optional): Directory that saves the timeseries files. Defaults to "input/timeseries_data".
        date_begin (str, optional): Measurements prior this date are ignored. Defaults to "01.10.22 00:00".
    """
    makedirs(output_dir, exist_ok=True)
    date_begin = pytz.timezone("Europe/Berlin").localize(datetime.strptime(date_begin, "%d.%m.%y %H:%M"))
    ts_path_list = wr.s3.list_objects(bucket_dir, last_modified_begin=date_begin)
    ts_fn_list = [basename(ts_path) for ts_path in ts_path_list]

    print("file exists      --> ---", "\nfile downloading --> +++\n")
    for fn in ts_fn_list:
        if exists(join(output_dir, fn)):
            print("---", fn)
            continue
        else:
            print("+++", fn)
            input_fn = join(bucket_dir, fn)
            output_fn = join(output_dir, fn)
            try:
                wr.s3.download(input_fn, output_fn)
            except Exception as e:
                print(e)

    print(f"{'-' * 100}\nTotal amount of files in {output_dir}:", len(listdir(output_dir)))
