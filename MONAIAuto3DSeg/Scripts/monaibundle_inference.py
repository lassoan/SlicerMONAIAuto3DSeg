import logging
import shutil
from pathlib import Path

import fire
from monai.bundle import ConfigParser
import torch

from monai.config import print_config
print_config()

def main(bundle_root_dir, image_file, result_file):
    device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device(0)
    logging.info("Running inference on CPU...")
    try:
        _infer(bundle_root_dir, image_file, result_file, device)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if device == torch.device("cpu"):
            logging.error(e)
            raise

        logging.info("Retrying inference on CPU...")
        try:
            _infer(bundle_root_dir, image_file, result_file, torch.device("cpu"))
            logging.info("Successfully completed inference on CPU")
        except Exception as cpu_error:
            logging.error(f"CPU inference also failed: {cpu_error}")
            raise
    finally:
        torch.cuda.empty_cache()
    logging.info("Finished")


def _infer(bundle_root_dir, image_file, result_file, device):
    with torch.no_grad():
        image_path = Path(image_file)
        result_path = Path(result_file)
        bundle_root_path = Path(bundle_root_dir)
        dataset_dir = image_path.parent
        output_dir = result_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        configFilePath = bundle_root_path.joinpath("configs", "inference.json")

        configParser = ConfigParser()
        configParser.read_config(configFilePath.as_posix())
        configParser["bundle_root"] = bundle_root_path.as_posix()
        configParser["dataset_dir"] = dataset_dir.as_posix()
        configParser["output_dir"] = output_dir.as_posix()
        configParser["datalist"] = [image_path.as_posix()]
        configParser["datalist"] = [image_path.as_posix()]
        configParser["window_device"] = "cpu"
        configParser["device"] = device
        configParser["inferer"]["device"] = device
        # avoid hanging in windows
        configParser["dataloader"]["num_workers"] = 0
        assert len(configParser["datalist"]) > 0, "Empty data list as input"
        dataloader = configParser.get_parsed_content("dataloader")
        assert len(dataloader) > 0
        evaluator = configParser.get_parsed_content("evaluator")
        evaluator.run()

        bundle_output = _find_output_file(output_dir)
        logging.info(f"Moving output from {bundle_output} to {result_path}")
        shutil.move(str(bundle_output), str(result_path))
        logging.info(f"Output file created: {result_path}")


def _find_output_file(output_dir: Path):
    files = list(output_dir.rglob("*.nii.gz",))
    if files:
        # Return the most recently modified file
        return max(files, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No output file found in {output_dir}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    fire.Fire(main)