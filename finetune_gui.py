import gradio as gr
import json
import math
import os
import subprocess


def save_variables(
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_model,
    # model_list,
    train_dir,
    image_folder,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    train_text_encoder,
    convert_to_safetensors,
    convert_to_ckpt,
):
    # Return the values of the variables as a dictionary
    variables = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "v2": v2,
        "v_model": v_model,
        # "model_list": model_list,
        "train_dir": train_dir,
        "image_folder": image_folder,
        "output_dir": output_dir,
        "max_resolution": max_resolution,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "lr_warmup": lr_warmup,
        "dataset_repeats": dataset_repeats,
        "train_batch_size": train_batch_size,
        "epoch": epoch,
        "save_every_n_epochs": save_every_n_epochs,
        "mixed_precision": mixed_precision,
        "save_precision": save_precision,
        "seed": seed,
        "num_cpu_threads_per_process": num_cpu_threads_per_process,
        "train_text_encoder": train_text_encoder,
        "convert_to_safetensors": convert_to_safetensors,
        "convert_to_ckpt": convert_to_ckpt,
    }

    # Save the data to the selected file
    with open(file_path, "w") as file:
        json.dump(variables, file)


def load_variables(file_path):
    # load variables from JSON file
    with open(file_path, "r") as f:
        my_data = json.load(f)

    # Return the values of the variables as a dictionary
    return (
        my_data["pretrained_model_name_or_path"],
        my_data["v2"],
        my_data["v_model"],
        my_data["train_dir"],
        # my_data["model_list"],
        my_data["image_folder"],
        my_data["output_dir"],
        my_data["max_resolution"],
        my_data["learning_rate"],
        my_data["lr_scheduler"],
        my_data["lr_warmup"],
        my_data["dataset_repeats"],
        my_data["train_batch_size"],
        my_data["epoch"],
        my_data["save_every_n_epochs"],
        my_data["mixed_precision"],
        my_data["save_precision"],
        my_data["seed"],
        my_data["num_cpu_threads_per_process"],
        my_data["train_text_encoder"],
        my_data["convert_to_safetensors"],
        my_data["convert_to_ckpt"],
    )


def train_model(
    create_caption,
    create_buckets,
    train,
    pretrained_model_name_or_path,
    v2,
    v_model,
    train_dir,
    image_folder,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    train_text_encoder,
    convert_to_safetensors,
    convert_to_ckpt,
):
    # create caption json file
    if create_caption:
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        command = [
            "./venv/Scripts/python.exe",
            "script/merge_captions_to_metadata.py",
            "--caption_extension",
            ".txt",
            image_folder,
            "{}/meta_cap.json".format(train_dir),
        ]

        # Run the command
        subprocess.run(command)

    # create images buckets
    if create_buckets:
        command = [
            "./venv/Scripts/python.exe",
            "script/prepare_buckets_latents.py",
            image_folder,
            "{}/meta_cap.json".format(train_dir),
            "{}/meta_lat.json".format(train_dir),
            "--full_path",
            pretrained_model_name_or_path,
            "--batch_size",
            "4",
            "--max_resolution",
            max_resolution,
            "--mixed_precision",
            mixed_precision,
        ]

        # Run the command
        subprocess.run(command)

    if train:
        image_num = len([f for f in os.listdir(image_folder) if f.endswith(".npz")])
        print(f"image_num = {image_num}")

        repeats = int(image_num) * int(dataset_repeats)
        print(f"repeats = {str(repeats)}")

        # calculate max_train_steps
        max_train_steps = int(
            math.ceil(float(repeats) / int(train_batch_size) * int(epoch))
        )
        print(f"max_train_steps = {max_train_steps}")

        lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
        print(f"lr_warmup_steps = {lr_warmup_steps}")

        # print(f"v2: {v2}, v_model: {v_model}, train_text_encoder: {train_text_encoder}")
        # v2_parm = "--v2" if v2 else '""
        # v_model_parm = "--v_parameterization" if v_model else '""
        # train_text_encoder_parm = "--train_text_encoder" if train_text_encoder else '""

        # print(
        #     f"v2: {v2_parm}, v_model: {v_model_parm}, train_text_encoder: {train_text_encoder_parm}"
        # )

        run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "script/fine_tune.py"'
        if v2:
            run_cmd += " --v2"
        if v_model:
            run_cmd += " --v_parameterization"
        if train_text_encoder:
            run_cmd += " --train_text_encoder"
        run_cmd += f" --pretrained_model_name_or_path={pretrained_model_name_or_path}"
        run_cmd += f" --in_json={train_dir}/meta_lat.json"
        run_cmd += f" --train_data_dir={image_folder}"
        run_cmd += f" --output_dir={output_dir}"
        run_cmd += f" --train_batch_size={train_batch_size}"
        run_cmd += f" --dataset_repeats={dataset_repeats}"
        run_cmd += f" --learning_rate={learning_rate}"
        run_cmd += f" --lr_scheduler={lr_scheduler}"
        run_cmd += f" --lr_warmup_steps={lr_warmup_steps}"
        run_cmd += f" --max_train_steps={max_train_steps}"
        run_cmd += f" --use_8bit_adam"
        run_cmd += f" --xformers"
        run_cmd += f" --mixed_precision={mixed_precision}"
        run_cmd += f" --save_every_n_epochs={save_every_n_epochs}"
        run_cmd += f" --seed={seed}"
        run_cmd += f" --save_precision={save_precision}"

        # print(run_cmd)

        # command = [
        #     "accelerate",
        #     "launch",
        #     f"--num_cpu_threads_per_process={num_cpu_threads_per_process}",
        #     f"script/fine_tune.py",
        #     f"{v2_parm}",
        #     f"{v_model_parm}",
        #     f"{train_text_encoder_parm}",
        #     f"--pretrained_model_name_or_path={pretrained_model_name_or_path}",
        #     f"--in_json={train_dir}/meta_lat.json",
        #     f"--train_data_dir={image_folder}",
        #     f"--output_dir={output_dir}",
        #     f"--train_batch_size={train_batch_size}",
        #     f"--dataset_repeats={dataset_repeats}",
        #     f"--learning_rate={learning_rate}",
        #     f"--lr_scheduler={lr_scheduler}",
        #     f"--lr_warmup_steps={lr_warmup_steps}",
        #     f"--max_train_steps={max_train_steps}",
        #     f"--use_8bit_adam",
        #     f"--xformers",
        #     f"--mixed_precision={mixed_precision}",
        #     f"--save_every_n_epochs={save_every_n_epochs}",
        #     f"--seed={seed}",
        #     f"--save_precision={save_precision}",
        # ]

        # Run the command
        subprocess.run(run_cmd)

    # Return the values of the variables as a dictionary
    # return


def set_pretrained_model_name_or_path_input(value, v2, v_model):
    # define a list of substrings to search for
    substrings_v2 = ["stable-diffusion-2-1-base", "stable-diffusion-2-base"]

    # check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
    if str(value) in substrings_v2:
        print("SD v2 model detected. Setting --v2 parameter")
        v2 = True
        v_model = False
        value = "stabilityai/{}".format(value)

        return value, v2, v_model

    # define a list of substrings to search for v-objective
    substrings_v_model = ["stable-diffusion-2-1", "stable-diffusion-2"]

    # check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_model list
    if str(value) in substrings_v_model:
        print("SD v2 v_model detected. Setting --v2 parameter and --v_parameterization")
        v2 = True
        v_model = True
        value = "stabilityai/{}".format(value)

        return value, v2, v_model

    if value == "custom":
        value = "<enter path to custom model or name of pretrained model>"
        v2 = False
        v_model = False

        return value, v2, v_model


# Define the output element
output = gr.outputs.Textbox(label="Values of variables")

interface = gr.Blocks()

with interface:
    gr.Markdown("Enter kohya finetuner parameter using this interface.")
    with gr.Accordion("Configuration File Load/Save", open=False):
        with gr.Row():
            config_file_name = gr.inputs.Textbox(label="Config file name", default="")
            b1 = gr.Button("Load config")
            b2 = gr.Button("Save config")
    with gr.Tab("model"):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path_input = gr.inputs.Textbox(
                label="Pretrained model name or path",
                default="<enter path to custom model or name of pretrained model>",
            )
            model_list = gr.Dropdown(
                label="Model Quick Pick",
                choices=[
                    "custom",
                    "stable-diffusion-2-1-base",
                    "stable-diffusion-2-base",
                    "stable-diffusion-2-1",
                    "stable-diffusion-2",
                ],
                value="custom",
            )
        with gr.Row():
            v2_input = gr.inputs.Checkbox(label="v2", default=True)
            v_model_input = gr.inputs.Checkbox(label="v_model", default=False)
        model_list.change(
            set_pretrained_model_name_or_path_input,
            inputs=[model_list, v2_input, v_model_input],
            outputs=[pretrained_model_name_or_path_input, v2_input, v_model_input],
        )
    with gr.Tab("training dataset and output directory"):
        train_dir_input = gr.inputs.Textbox(
            label="Train directory", default="D:\\models\\test\\samdoesart2"
        )
        image_folder_input = gr.inputs.Textbox(
            label="Image folder", default="D:\\dataset\\samdoesart2\\raw"
        )
        output_dir_input = gr.inputs.Textbox(
            label="Output directory",
            default="D:\\models\\test\\samdoesart2\\model_e2\\",
        )
        max_resolution_input = gr.inputs.Textbox(
            label="Max resolution", default="512,512"
        )
    with gr.Tab("training parameters"):
        with gr.Row():
            learning_rate_input = gr.inputs.Textbox(label="Learning rate", default=1e-6)
            lr_scheduler_input = gr.Dropdown(
                label="LR Scheduler",
                choices=[
                    "constant",
                    "constant_with_warmup",
                    "cosine",
                    "cosine_with_restarts",
                    "linear",
                    "polynomial",
                ],
                value="constant",
            )
            lr_warmup_input = gr.inputs.Textbox(label="LR warmup", default=0)
        with gr.Row():
            dataset_repeats_input = gr.inputs.Textbox(
                label="Dataset repeats", default=40
            )
            train_batch_size_input = gr.inputs.Textbox(
                label="Train batch size", default=1
            )
            epoch_input = gr.inputs.Textbox(label="Epoch", default=1)
        with gr.Row():
            save_every_n_epochs_input = gr.inputs.Textbox(
                label="Save every N epochs", default=1
            )
            mixed_precision_input = gr.Dropdown(
                label="Mixed precision",
                choices=[
                    "no",
                    "fp16",
                    "bf16",
                ],
                value="fp16",
            )
            save_precision_input = gr.Dropdown(
                label="Save precision",
                choices=[
                    "float",
                    "fp16",
                    "bf16",
                ],
                value="fp16",
            )
        with gr.Row():
            seed_input = gr.inputs.Textbox(label="Seed", default=1234)
            num_cpu_threads_per_process_input = gr.inputs.Textbox(
                label="Number of CPU threads per process", default=4
            )
            train_text_encoder_input = gr.inputs.Checkbox(
                label="Train text encoder", default=True
            )
    with gr.Tab("model conveersion"):
        convert_to_safetensors_input = gr.inputs.Checkbox(
            label="Convert to SafeTensors", default=False
        )
        convert_to_ckpt_input = gr.inputs.Checkbox(
            label="Convert to CKPT", default=False
        )
    # define the buttons

    with gr.Box():
        with gr.Row():
            create_caption = gr.inputs.Checkbox(label="Create Caption", default=True)
            create_buckets = gr.inputs.Checkbox(label="Create Buckets", default=True)
            train = gr.inputs.Checkbox(label="Train", default=True)
        b3 = gr.Button("Run")

    output = gr.outputs.Textbox(label="Values of variables")

    b1.click(
        load_variables,
        inputs=[config_file_name],
        outputs=[
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            # model_list,
            train_dir_input,
            image_folder_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            dataset_repeats_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            train_text_encoder_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
        ],
    )
    b2.click(
        save_variables,
        inputs=[
            config_file_name,
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            # model_list,
            train_dir_input,
            image_folder_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            dataset_repeats_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            train_text_encoder_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
        ],
        outputs=output,
    )
    b3.click(
        train_model,
        inputs=[
            create_caption,
            create_buckets,
            train,
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            train_dir_input,
            image_folder_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            dataset_repeats_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            train_text_encoder_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
        ],
        outputs=output,
    )


# # Create the interface
# interface = gr.Interface(
#     train_model,
#     [
#         pretrained_model_name_or_path_input,
#         v2_input,
#         v_model_input, model_list,
#         train_dir_input,
#         image_folder_input,
#         output_dir_input,
#         max_resolution_input,
#         learning_rate_input,
#         lr_scheduler_input,
#         lr_warmup_input,
#         dataset_repeats_input,
#         train_batch_size_input,
#         epoch_input,
#         save_every_n_epochs_input,
#         mixed_precision_input,
#         save_precision_input,
#         seed_input,
#         num_cpu_threads_per_process_input,
#         train_text_encoder_input,
#         convert_to_safetensors_input,
#         convert_to_ckpt_input
#     ],
#     outputs="text"
# )

# Show the interface
interface.launch()

# # Get the values of the input variables
# variable_values = train_model()

# # Display the values in the output textbox
# output.value = str(variable_values)
