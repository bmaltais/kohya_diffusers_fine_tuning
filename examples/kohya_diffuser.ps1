# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-emaonly.ckpt"
# $pretrained_model_name_or_path = "D:\models\f222\f222.ckpt"
# $pretrained_model_name_or_path = "D:\models\retrofuturism_pinup\bettie_page_diffusers_fine_tuned_e2\last.ckpt"
$train_dir = "D:\models\test\kohya_ft"
$image_folder = "D:\dataset\vintage\raw2"
$output_dir = "D:\models\test\kohya_ft\model\"
$repo_path = "D:\kohya_diffusers_fine_tuning"

$learning_rate = 5e-6
$lr_scheduler = "polynomial" # Default is constant
$lr_warmup = 10 # % of steps to warmup for 0 - 100. Default is 0.
$dataset_repeats = 150
$train_batch_size = 8
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
if (!(Test-Path -Path $train_dir))
{
    New-Item -Path $train_dir -ItemType "directory"
}

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

$lr_warmup_steps = [Math]::Round($lr_warmup * $max_train_set / 100)
Write-Host("lr_warmup_steps = $lr_warmup_steps")

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process $repo_path\script\fine_tune.py `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir="$image_folder" `
    --output_dir=$output_dir `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --lr_scheduler=$lr_scheduler `
    --lr_warmup_steps=$lr_warmup_steps `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --seed=494481440 `
    --train_text_encoder `
    --save_precision=$save_precision