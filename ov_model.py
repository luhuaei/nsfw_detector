from openvino.runtime import Core
import openvino.runtime.properties as props
import openvino.runtime.properties.hint as hints
from pathlib import Path
import time
import threading
import functools
import os
from transformers import ViTImageProcessor, AutoConfig
import torch

# constant
ROOT_CACHE_PATH = Path(os.getenv("ROOT_CACHE_PATH", "/lzcapp/cache/"))
OV_MODEL_CACHE_PATH = ROOT_CACHE_PATH / "ov_model"

# 默认使用 CPU
OV_DEVICE_NAME = "CPU"

# init openvino core
core = Core()
print("gpu", core.available_devices)

for d in core.available_devices:
    if d == "GPU":
        OV_DEVICE_NAME = d
print("choose device: ",OV_DEVICE_NAME)

core.set_property({
    props.cache_dir(): OV_MODEL_CACHE_PATH,
    props.enable_mmap(): True,
    # props.compilation_num_threads(): 1,
})

# 使用装饰器简单实现了一个 scale to zero 的版本，在指定时间内没有任务访问，则把模型从内存中卸掉
class AutoUnloadDecorator:
    def __init__(self, model_name, timeout=300):  # timeout单位为秒
        self.model_name = model_name
        self.timeout = timeout
        self.last_access = time.time()
        self.model = None
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor_and_unload)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_and_unload(self):
        while True:
            time.sleep(self.timeout / 2)  # 检查频率为timeout的一半
            with self.lock:
                if time.time() - self.last_access > self.timeout:
                    if self.model is not None:
                        print(f"{self.model_name} No access in last {self.timeout} seconds. Unloading model.")
                        del self.model
                        self.model = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with self.lock:
                self.last_access = time.time()
                if self.model is None:
                    begin = time.time()
                    print(f"Loading {self.model_name} model...")
                    self.model = func(*args, **kwargs)
                    print(f"Loading {self.model_name} done. spend time: {time.time() - begin} seconds.")
                return self.model
        return wrapped

@AutoUnloadDecorator(model_name="nsfw_detection")
def load_nsfw_detection_model():
    return core.compile_model(
        Path("./Falconsai_nsfw_image_detection_ov_fp16/openvino_model.xml"),
        device_name=OV_DEVICE_NAME)

def load_nsfw_detection_preprocess():
    processor = ViTImageProcessor.from_pretrained('./Falconsai_nsfw_image_detection_ov_fp16')
    return processor


class NSFW():
    def __init__(self):
        self.config = AutoConfig.from_pretrained("./Falconsai_nsfw_image_detection_ov_fp16")
        self.id2label = self.config.id2label
        self.preprocess = load_nsfw_detection_preprocess()

    def __call__(self, images):
        model = load_nsfw_detection_model()
        inputs = self.preprocess(images=images, return_tensors="pt")
        outputs = model(inputs["pixel_values"])
        # Apply softmax to get probabilities
        logits = torch.tensor(outputs[model.outputs[0]])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        results = []
        for prob in probs:
            results.append({
                "normal": float(prob[0]),
                "nsfw": float(prob[1]),
            })
        return results
