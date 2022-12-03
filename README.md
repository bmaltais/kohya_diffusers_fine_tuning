# Diffusers Fine Tuning

This subfolder provide all the required tools  to run the diffusers fine tuning version found in this note: https://note.com/kohya_ss/n/nbf7ce8d80f29

## Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/bmaltais/kohya_diffusers_fine_tuning.git
cd kohya_diffusers_fine_tuning

python -m venv --system-site-packages venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config

```

Answers to accelerate config:

```txt
- 0
- 0
- NO
- NO
- All
- fp16
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd kohya_script
git pull
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Folders configuration

Simply put all the images you will want to train on in a single directory. It does not matter what size or aspect ratio they have. It is your choice.

## Captions

Each file need to be accompanied by a caption file describing what the image is about. FOr example, if you want to train on cute dog pictures you can put `cute dog` as the caption in every file. You can use the `tools\caption.ps1` sample code to help out with that:

```powershell
$folder = "sample"
$file_pattern="*.*"
$caption_text="cute dog"

$files = Get-ChildItem "$folder\$file_pattern" -Include *.png, *.jpg, *.webp -File
foreach ($file in $files) {
    if (-not(Test-Path -Path $folder\"$($file.BaseName).txt" -PathType Leaf)) {
        New-Item -ItemType file -Path $folder -Name "$($file.BaseName).txt" -Value $caption_text
    }
}
```

## Execution

### SD1.5 example

Edit and paste the following in a Powershell terminal:

```powershell
# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-mse-vae.ckpt"
$train_dir = "D:\models\cute_dog"
$image_folder = "D:\models\cute_dog\images"
$output_dir = "D:\models\cute_dog\diffusers_fine_tuned_model_bucket"
$repo_path = "D:\kohya_diffusers_fine_tuning"

$learning_rate = 1e-6
$dataset_repeats = 50
$train_batch_size = 1
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="bf16"
$save_precision="bf16"
$num_cpu_threads_per_process=6

$max_resolution = "512,512"

# stop script on error
$ErrorActionPreference = "Stop"

# activate venv
cd $repo_path
.\venv\Scripts\activate

# create caption json file
python $repo_path\script\merge_captions_to_metadata.py `
--caption_extention ".txt" $image_folder $train_dir"\meta_cap.json"

# create images buckets
python $repo_path\script\prepare_buckets_latents.py `
    $image_folder `
    $train_dir"\meta_cap.json" `
    $train_dir"\meta_lat.json" `
    $pretrained_model_name_or_path `
    --batch_size 4 --max_resolution $max_resolution --mixed_precision $mixed_precision

# Get number of valid images
$image_num = Get-ChildItem "$image_folder" -Recurse -File -Include *.npz | Measure-Object | %{$_.Count}

$repeats = $image_num * $dataset_repeats
Write-Host("Repeats = $repeats")

# calculate max_train_set
$max_train_set = [Math]::Ceiling($repeats / $train_batch_size * $epoch)
Write-Host("max_train_set = $max_train_set")

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process $repo_path\script\fine_tune.py `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir="$image_folder" `
    --output_dir=$output_dir `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --seed=494481440 `
    --train_text_encoder `
    --save_precision=$save_precision

```

### SD2.0 512 Base example

```powershell
# variable values
$pretrained_model_name_or_path = "D:\models\v20-512.ckpt"
$train_dir = "D:\models\cute_dog"
$image_folder = "D:\models\cute_dog\images"
$output_dir = "D:\models\cute_dog\diffusers_fine_tuned_model_bucket"
$repo_path = "D:\kohya_diffusers_fine_tuning"

$learning_rate = 1e-6
$dataset_repeats = 50
$train_batch_size = 1
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="bf16"
$save_precision="bf16"
$num_cpu_threads_per_process=6

$max_resolution = "512,512"

# stop script on error
$ErrorActionPreference = "Stop"

# activate venv
cd $repo_path
.\venv\Scripts\activate

# create caption json file
python $repo_path\script\merge_captions_to_metadata.py `
--caption_extention ".txt" $image_folder $train_dir"\meta_cap.json"

# create images buckets
python $repo_path\script\prepare_buckets_latents.py `
    $image_folder `
    $train_dir"\meta_cap.json" `
    $train_dir"\meta_lat.json" `
    $pretrained_model_name_or_path `
    --batch_size 4 --max_resolution $max_resolution --mixed_precision $mixed_precision

# Get number of valid images
$image_num = Get-ChildItem "$image_folder" -Recurse -File -Include *.npz | Measure-Object | %{$_.Count}

$repeats = $image_num * $dataset_repeats
Write-Host("Repeats = $repeats")

# calculate max_train_set
$max_train_set = [Math]::Ceiling($repeats / $train_batch_size * $epoch)
Write-Host("max_train_set = $max_train_set")

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process $repo_path\script\fine_tune.py `
    --v2 `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir="$image_folder" `
    --output_dir=$output_dir `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --seed=494481440 `
    --train_text_encoder `
    --save_precision=$save_precision

# Add the inference yaml file along with the model for proper loading. Need to have the same name as model... Most likelly "last.yaml" in our case.
cp v2_inference\v2-inference.yaml $output_dir"\last.yaml"

```

### SD2.0 768v Base example

```powershell
# variable values
$pretrained_model_name_or_path = "D:\models\v20-768v.ckpt"
$train_dir = "D:\models\cute_dog"
$image_folder = "D:\models\cute_dog\images"
$output_dir = "D:\models\cute_dog\diffusers_fine_tuned_model_bucket"
$repo_path = "D:\kohya_diffusers_fine_tuning"

$learning_rate = 1e-6
$dataset_repeats = 50
$train_batch_size = 1
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="bf16"
$save_precision="bf16"
$num_cpu_threads_per_process=6

$max_resolution = "768,768"

# stop script on error
$ErrorActionPreference = "Stop"

# activate venv
cd $repo_path
.\venv\Scripts\activate

# create caption json file
python $repo_path\script\merge_captions_to_metadata.py `
--caption_extention ".txt" $image_folder $train_dir"\meta_cap.json"

# create images buckets
python $repo_path\script\prepare_buckets_latents.py `
    $image_folder `
    $train_dir"\meta_cap.json" `
    $train_dir"\meta_lat.json" `
    $pretrained_model_name_or_path `
    --batch_size 4 --max_resolution $max_resolution --mixed_precision $mixed_precision

# Get number of valid images
$image_num = Get-ChildItem "$image_folder" -Recurse -File -Include *.npz | Measure-Object | %{$_.Count}

$repeats = $image_num * $dataset_repeats
Write-Host("Repeats = $repeats")

# calculate max_train_set
$max_train_set = [Math]::Ceiling($repeats / $train_batch_size * $epoch)
Write-Host("max_train_set = $max_train_set")

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process $repo_path\script\fine_tune.py `
    --v2 `
    --v_parameterization `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir="$image_folder" `
    --output_dir=$output_dir `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --seed=494481440 `
    --train_text_encoder `
    --save_precision=$save_precision

# Add the inference yaml file along with the model for proper loading. Need to have the same name as model... Most likelly "last.yaml" in our case.
cp v2_inference\v2-inference-v.yaml $output_dir"\last.yaml"
```

## Options list

```txt

```

## Change history

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
