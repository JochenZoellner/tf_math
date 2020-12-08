import argparse
import json
import logging
import os
import glob
import pathlib
import pandas
logger = logging.getLogger(os.path.basename(__file__))



def main(args):
    logger.info(f"Running main() of {__file__}")
    logger.info(f"Root_dir: {args.root_dir}")
    if str(args.root_dir).startswith("[") and str(args.root_dir).endswith("]"):
        dir_list = str(args.root_dir)[1:-1].split(",")
    else:
        dir_list = [args.root_dir]
    file_list = []
    for dir in dir_list:
        file_list.extend([str(x) for x in pathlib.Path(dir).rglob("lav-*.json")])
    logger.info(f"Files found:{chr(10)}  {(chr(10) + '  ').join(file_list)}")
    if len(file_list) != len(set([os.path.basename(x) for x in file_list])):
        # check if there are the same checkpoint id in an other dir and exit
        # avoid duplicated id's which would result in overwriting a row
        raise AttributeError("Duplicated ID's found, exit!")
    fn_dict_list = {}
    sorted(file_list)
    for fn in file_list:
        with open(fn, "r") as fp_json:
            json_str = fp_json.read()
            fn_dict = json.loads(json_str)
        logger.debug(f'{os.path.basename(fn) + chr(10) + json_str}')
        fn_dict_list[os.path.basename(fn)[4:-5]] = fn_dict
    # join header
    header = []
    for fn_key in fn_dict_list:
        for metric_key in fn_dict_list[fn_key]:
            if metric_key not in header:
                header.append(metric_key)
    logger.info(f'\n{chr(10).join(["  "  + x for x in header])}')

    df_list = [pandas.DataFrame({file_key: fn_dict_list[file_key]}).transpose() for file_key in fn_dict_list]

    data_frame = pandas.concat(df_list)
    data_frame.to_csv(f"lav_collection-{'_'.join([os.path.basename(x) for x in dir_list])}.csv", sep="\t")

    print("Result", data_frame)
    # # for metric_key in value:
    # #     # data_frame.loc[[key], [metric_key]] = value[metric_key]
    # data_frame.from_dict(value)

    # data_frame[]
    pass


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{__file__}'")
    parser.add_argument("--root_dir", type=str, help="set dir to search recusivley for lav-*.json files, "
                                                     "accepts multi dirs like [dir1,dir2]")
    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    # logger.setLevel("INFO")
    logger.setLevel("DEBUG")
    logger.info(f"Running {__file__} as __main__...")
    arguments = parse_args()
    main(args=arguments)
