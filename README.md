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

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed a bit!!!

Due to the filesize I can't host the DLLs needed for CUDNN 8.6 on Github, I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090) you can download them from here: https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip

To install simply unzip the directory and place the cudnn_windows folder in the root of the kohya_diffusers_fine_tuning repo.

Run the following command to install:

```
python cudann_1.8_install.py
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd kohya_diffusers_fine_tuning
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

## GUI

There is now support for GUI based training using gradio. You can start the GUI interface by running:

```powershell
python .\dreambooth_gui.py
```

## Manual cli Execution

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
usage: fine_tune.py [-h] [--v2] [--v_parameterization] [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                    [--in_json IN_JSON] [--shuffle_caption] [--train_data_dir TRAIN_DATA_DIR]
                    [--dataset_repeats DATASET_REPEATS] [--output_dir OUTPUT_DIR] [--use_safetensors]
                    [--train_text_encoder] [--hypernetwork_module HYPERNETWORK_MODULE]
                    [--hypernetwork_weights HYPERNETWORK_WEIGHTS] [--save_every_n_epochs SAVE_EVERY_N_EPOCHS]
                    [--save_state] [--resume RESUME] [--max_token_length {None,150,225}]
                    [--train_batch_size TRAIN_BATCH_SIZE] [--use_8bit_adam] [--mem_eff_attn] [--xformers]
                    [--diffusers_xformers] [--learning_rate LEARNING_RATE] [--max_train_steps MAX_TRAIN_STEPS]
                    [--seed SEED] [--gradient_checkpointing]
                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--mixed_precision {no,fp16,bf16}]      
                    [--save_precision {None,float,fp16,bf16}] [--clip_skip CLIP_SKIP] [--debug_dataset]
                    [--logging_dir LOGGING_DIR] [--log_prefix LOG_PREFIX] [--lr_scheduler LR_SCHEDULER]
                    [--lr_warmup_steps LR_WARMUP_STEPS]

options:
  -h, --help            show this help message and exit
  --v2                  load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む
  --v_parameterization  enable v-parameterization training / v-parameterization学習を有効にする
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint /
                        学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル
  --in_json IN_JSON     metadata file to input / 読みこむメタデータファイル
  --shuffle_caption     shuffle comma-separated caption when fine tuning / fine tuning時にコンマで区切られたcaptionの各 要素をshuffleする
  --train_data_dir TRAIN_DATA_DIR
                        directory for train images / 学習画像データのディレクトリ
  --dataset_repeats DATASET_REPEATS
                        num times to repeat dataset / 学習にデータセットを繰り返す回数
  --output_dir OUTPUT_DIR
                        directory to output trained model, save as same format as input / 学習後のモデル出力先ディレクトリ（入力と同じ形式で保存）
  --use_safetensors     use safetensors format to save / checkpoint、モデルをsafetensors形式で保存する
  --train_text_encoder  train text encoder / text encoderも学習する
  --hypernetwork_module HYPERNETWORK_MODULE
                        train hypernetwork instead of fine tuning, module to use / fine
                        tuningの代わりにHypernetworkの学習をする場合、そのモジュール
  --hypernetwork_weights HYPERNETWORK_WEIGHTS
                        hypernetwork weights to initialize for additional training /
                        Hypernetworkの学習時に読み込む重み（Hypernetworkの追加学習）
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する
  --save_state          save training state additionally (including optimizer states etc.) /
                        optimizerなど学習状態も含めたstateを追加で保存する
  --resume RESUME       saved state to resume training / 学習再開するモデルのstate
  --max_token_length {None,150,225}
                        max token length of text encoder (default for 75, 150 or 225) / text
                        encoderのトークンの最大長（未指定で75、150または225が指定可）
  --train_batch_size TRAIN_BATCH_SIZE
                        batch size for training / 学習時のバッチサイズ
  --use_8bit_adam       use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesの インストールが必要）
  --mem_eff_attn        use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う   
  --xformers            use xformers for CrossAttention / CrossAttentionにxformersを使う
  --diffusers_xformers  use xformers by diffusers (Hypernetworks doesn't work) /
                        Diffusersでxformersを使用する（Hypernetwork利用不可）
  --learning_rate LEARNING_RATE
                        learning rate / 学習率
  --max_train_steps MAX_TRAIN_STEPS
                        training steps / 学習ステップ数
  --seed SEED           random seed for training / 学習時の乱数のseed
  --gradient_checkpointing
                        enable gradient checkpointing / grandient checkpointingを有効にする
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass /
                        学習時に逆伝播をする前に勾配を合計するステップ数
  --mixed_precision {no,fp16,bf16}
                        use mixed precision / 混合精度を使う場合、その精度
  --save_precision {None,float,fp16,bf16}
                        precision in saving (available in StableDiffusion checkpoint) /
                        保存時に精度を変更して保存する（StableDiffusion形式での保存時のみ有効）
  --clip_skip CLIP_SKIP
                        use output of nth layer from back of text encoder (n>=1) / text
                        encoderの後ろからn番目の層の出力を用いる（nは1以上）
  --debug_dataset       show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない ）
  --logging_dir LOGGING_DIR
                        enable logging and output TensorBoard log to this directory /
                        ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する
  --log_prefix LOG_PREFIX
                        add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列
  --lr_scheduler LR_SCHEDULER
                        scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts,
                        polynomial, constant (default), constant_with_warmup
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler (default is 0) /
                        学習率のスケジューラをウォームアップするステップ数（デフォルト0）
```

## Change history

* 12/13 (v8):
    - WD14Tagger now works on its own.
    - Added support for learning to fp16 up to the gradient. Go to "Building the environment and preparing scripts for Diffusers for more info".
* 12/10 (v7):
    - We have added support for Diffusers 0.10.2.
    - In addition, we have made other fixes.
    - For more information, please see the section on "Building the environment and preparing scripts for Diffusers" in our documentation.
* 12/6 (v6): We have responded to reports that some models experience an error when saving in SafeTensors format.
* 12/5 (v5):
    - .safetensors format is now supported. Install SafeTensors as "pip install safetensors". When loading, it is automatically determined by extension. Specify use_safetensors options when saving.
    - Added an option to add any string before the date and time log directory name log_prefix.
    - Cleaning scripts now work without either captions or tags.
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
