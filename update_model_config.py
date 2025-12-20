import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model configuration file.")
    parser.add_argument(
        "model_path", type=str, help="Path to the model configuration JSON file."
    )

    args = parser.parse_args()

    # Load existing configuration
    with open(os.path.join(args.model_path, "config.json"), "r") as config_file:
        config = json.load(config_file)

    with open(os.path.join(args.model_path, "config.bak.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    config["model_type"] = "qwen3_omni_moe"
    thinker_config = config["thinker_config"]
    thinker_config["model_type"] = "qwen3_omni_moe_thinker"

    thinker_config["audio_token_id"] = thinker_config[
        "audio_token_index"
    ]
    thinker_config["image_token_id"] = thinker_config[
        "image_token_index"
    ]
    thinker_config["audio_config"]["conv_chunksize"] = 500
    thinker_config["audio_config"]["model_type"] = "qwen3_omni_moe_audio_encoder"
    thinker_config["audio_config"]["downsample_hidden_size"] = 480

    thinker_config["text_config"]["rope_scaling"]["mrope_interleaved"] = True
    thinker_config["text_config"]["qkv_bias"] = False
    thinker_config["text_config"]["model_type"] = "qwen3_omni_moe_text"

    thinker_config["video_token_id"] = thinker_config["video_token_index"]
    thinker_config["vision_config"]["deepstack_visual_indexes"] = thinker_config[
        "vision_config"
    ]["deepstack_visual_multiscale_indexes"]
    thinker_config["vision_config"]["model_type"] = "qwen3_omni_moe_vision_encoder"
    thinker_config["vision_config"]["visual_multiscale_indexes"] = thinker_config[
        "vision_config"
    ]["deepstack_visual_multiscale_indexes"]
    thinker_config["vision_config"]["image_size"] = thinker_config["vision_config"][
        "img_size"
    ]
    # Save updated configuration back to file
    with open(os.path.join(args.model_path, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)
