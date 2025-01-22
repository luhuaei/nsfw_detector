convert:
	huggingface-cli download Falconsai/nsfw_image_detection --local-dir Falconsai/nsfw_image_detection
	optimum-cli export openvino --model ./Falconsai/nsfw_image_detection --weight-format fp16 --task image-classification Falconsai_nsfw_image_detection_ov_fp16

build_ov_image:
	docker build -f dockerfile -t nsfw_detector_ov:latest .
