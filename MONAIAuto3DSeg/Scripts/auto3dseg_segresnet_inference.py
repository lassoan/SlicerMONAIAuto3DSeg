import os
import numpy as np
import fire
import torch

import nibabel as nib
from monai.bundle import ConfigParser
from monai.data import decollate_batch, list_data_collate
from monai.utils import ImageMetaKey, convert_to_dst_type, optional_import, set_determinism
from monai.utils import MetaKeys

from torch.cuda.amp import GradScaler, autocast
from monai.inferers import SlidingWindowInfererAdapt


from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    ClassesToIndicesd,
    Compose,
    CropForegroundd,
    EnsureTyped,
    Identityd,
    Invertd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToDeviced,
    Orientationd,
)


def logits2pred(logits, sigmoid=False, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    if sigmoid:
        pred = torch.sigmoid(logits)
        pred = (pred >= 0.5)
    else:
        pred = torch.softmax(logits, dim=dim, dtype=torch.double).float()
        pred = torch.argmax(pred, dim=dim, keepdim=True).to(dtype=torch.uint8)            

    return pred


@torch.no_grad()
def main(model_file='model.pt',  image_file=None,  result_file = 'result.nii.gz', save_mode = None, **kwargs):


    if image_file is None or not os.path.exists(image_file):
        raise ValueError('Incorrect image filename:'+str(image_file))

    if not os.path.exists(model_file):
        raise ValueError('Cannot find model file:'+str(model_file))

    checkpoint = torch.load(model_file, map_location="cpu")

    if 'config' not in checkpoint:
        raise ValueError('Config not found in checkpoint (not a auto3dseg/segresnet model):'+str(model_file))

    config = checkpoint["config"]
    print(config)

    state_dict = checkpoint["state_dict"]

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0)
    sigmoid = config.get("sigmoid", False)


    model = ConfigParser(config["network"]).get_parsed_content()
    model.load_state_dict(state_dict, strict=True)

    print(f'Model epoch {epoch} metric {best_metric}')


    device=torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device(0)
    model = model.to(device=device, memory_format = torch.channels_last_3d) #gpu
    model.eval()


    # make input Transform chain
    ts = [ 
            LoadImaged(keys="image", ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=False),
            EnsureTyped(keys="image", data_type="tensor", dtype=torch.float, allow_missing_keys=True),
    ]

    if config.get("orientation_ras", False):
        print('Using orientation_ras')
        ts.append(Orientationd(keys="image", axcodes="RAS")) #reorient
    if config.get("crop_foreground", True):
        print('Using crop_foreground')
        ts.append(CropForegroundd(keys="image", source_key="image", margin=10, allow_smaller=True)) #subcrop

    if config.get("resample_resolution", None) is not None:
        pixdim = list(config["resample_resolution"])
        print(f'Using resample with  resample_resolution {pixdim}')

        ts.append(
            Spacingd(
                keys=["image"],
                pixdim=list(pixdim),
                mode=["bilinear"],
                dtype=torch.float,
                min_pixdim=np.array(pixdim) * 0.75,
                max_pixdim=np.array(pixdim) * 1.25,
                allow_missing_keys=True,
            )
        )    

    normalize_mode = config["normalize_mode"]
    intensity_bounds = config["intensity_bounds"]

    if normalize_mode == "none":
        pass
    elif normalize_mode in ["range", "ct"]:
        ts.append( ScaleIntensityRanged(keys="image", a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=-1, b_max=1, clip=False))
        ts.append(Lambdad(keys="image", func=lambda x: torch.sigmoid(x)))
    elif normalize_mode in ["meanstd", "mri"]:
        ts.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    elif normalize_mode in ["meanstdtanh"]:
        ts.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
        ts.append(Lambdad(keys="image", func=lambda x: 3*torch.tanh(x/3)))
    elif normalize_mode in ["pet"]:
        ts.append(Lambdad(keys="image", func=lambda x: torch.sigmoid((x - x.min()) / x.std())))
    else:
        raise ValueError("Unsupported normalize_mode" + str(normalize_mode))

 
    inf_transform = Compose(ts)

    # sliding_inferrer
    roi_size = config["roi_size"]
    # roi_size = [224, 224, 144]
    sliding_inferrer = SlidingWindowInfererAdapt(roi_size=roi_size, sw_batch_size=1, overlap=0.625, mode="gaussian", cache_roi_weight_map=False, progress=True)
    

    # process DATA
    batch_data = inf_transform([{"image" : image_file}])
    #original_affine = batch_data[0]['image_meta_dict']['original_affine']
    original_affine = batch_data[0]["image"].meta[MetaKeys.ORIGINAL_AFFINE]
    batch_data = list_data_collate([batch_data])
    data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=torch.channels_last_3d, device=device)

    print('Running Inference ...')
    with autocast(enabled=True):
        logits = sliding_inferrer(inputs=data, network=model)

    print(f"Logits {logits.shape}")
    # logits -> preds
    print('Converting logits into predictions')
    try:
        pred = logits2pred(logits, sigmoid=sigmoid)
    except RuntimeError as e:
        if not logits.is_cuda:
            raise e
        print(f"logits2pred failed on GPU pred retrying on CPU {logits.shape}")
        logits = logits.cpu()
        pred = logits2pred(logits, sigmoid=sigmoid)
    print(f"preds {pred.shape}")
    logits = None

    # pred = pred.cpu() # convert to CPU if the next step (reverse interpolation) is OOM on GPU
    # invert loading transforms (uncrop, reverse-resample, etc)
    post_transforms = Compose([Invertd(keys="pred", orig_keys="image", transform=inf_transform, nearest_interp=True)])

    batch_data["pred"] = convert_to_dst_type(pred, batch_data["image"], dtype=pred.dtype, device=pred.device)[0]  # make Meta tensor
    pred = [post_transforms(x)["pred"] for x in decollate_batch(batch_data)]
    seg = pred[0][0]
    print(f"preds inverted {seg.shape}")
    


    # special for kits and brats
    if save_mode == 'kits23' or 'kits23' in config['bundle_root']: #special case
        # convert 3 channel into 1 channel ints
        p2 = seg.any(0).to(dtype=torch.uint8)
        print(f'p2 step1 {p2.shape} {p2.dtype}')
        p2[seg[1:].any(0)] = 3
        print(f'p2 step2 {p2.shape} {p2.dtype}')

        p2[seg[2]] = 2
        print(f'p2 step3 {p2.shape} {p2.dtype}')

        seg = p2
        print(f"Updated seg for kits23 {seg.shape}")


    elif save_mode == 'brats23' or 'brats23' in config['bundle_root']: #special case brats
        # convert 3 channel into 1 channel ints
        p2 = 2 * seg.any(0).to(dtype=torch.uint8)
        p2[seg[1:].any(0)] = 1
        p2[seg[2]] = 3
        seg = p2
        print(f"Updated seg for brats23 {seg.shape}")

    seg = seg.cpu().numpy().astype(np.uint8)

    nib.save(nib.Nifti1Image(seg, affine = original_affine), result_file)
    print(f'ALL DONE, result saved in {result_file}')

if __name__ == '__main__':
  fire.Fire(main)
