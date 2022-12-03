# Diffusers Fine Tuning

This subfolder provide all the required tools  to run the diffusers fine tuning version found in this note: https://note.com/kohya_ss/n/nbf7ce8d80f29

## Releases

* 11/29 (v4):
    - DiffUsers 0.9.0 is required. Update as "pip install -U diffusers[torch]==0.9.0" in the virtual environment, and update the dependent libraries as "pip install --upgrade -r requirements.txt" if other errors occur.
    - Compatible with Stable Diffusion v2.0. Add the --v2 option when training (and pre-fetching latents). If you are using 768-v-ema.ckpt or stable-diffusion-2 instead of stable-diffusion-v2-base, add --v_parameterization as well when learning. Learn more about other options.
    - The minimum resolution and maximum resolution of the bucket can be specified when pre-fetching latents.
    - Corrected the calculation formula for loss (fixed that it was increasing according to the batch size).
    - Added options related to the learning rate scheduler.
    - So that you can download and learn DiffUsers models directly from Hugging Face. In addition, DiffUsers models can be saved during training.
    - Available even if the clean_captions_and_tags.py is only a caption or a tag.
    - Other minor fixes such as changing the arguments of the noise scheduler during training.
* 11/23 (v3):
    - Added WD14Tagger tagging script.
    - A log output function has been added to the fine_tune.py. Also, fixed the double shuffling of data.
    - Fixed misspelling of options for each script (caption_extention→caption_extension will work for the time being, even if it remains outdated).
