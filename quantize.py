import os import cv2 import numpy as np import onnx import vai_q_onnx import glob import random
# ================= 1. 定义数据读取器 =================
class YOLODataReader(vai_q_onnx.CalibrationDataReader): def __init__(self, calib_image_dir, input_name, shape): super().__init__()
        
        # 递归寻找 images 文件夹下的所有 jpg/png 图片
        search_path = os.path.join(calib_image_dir, '**', '*.*') all_files = glob.glob(search_path, recursive=True) self.image_paths = [f for f in all_files if 
        f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 量化不需要全部数据集，随机抽取 200 张即可
        if len(self.image_paths) > 200: random.seed(42) # 固定随机种子以保证可重复性 self.image_paths = random.sample(self.image_paths, 200)
            
        print(f"[INFO] 成功从 {calib_image_dir} 目录中选取了 {len(self.image_paths)} 张图片用于校准。")
        
        self.input_name = input_name self.shape = shape # (宽度, 高度) self.enum_data_dicts = iter(self._get_dataset()) def _get_dataset(self): for img_path in self.image_paths:
            # 读取图片
            img = cv2.imread(img_path) if img is None: continue
            
            # BGR 转 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整为模型所需的输入大小
            img = cv2.resize(img, self.shape)
            
            # 归一化 (YOLO通常是除以255.0，请确保与训练时一致)
            img = img.astype(np.float32) / 255.0
            
            # 将 HWC (高,宽,通道) 转换为 CHW (通道,高,宽)
            img = np.transpose(img, (2, 0, 1))
            
            # 增加 Batch 维度，形状变为 (1, C, H, W)
            img = np.expand_dims(img, axis=0)
            
            yield {self.input_name: img} def get_next(self): return next(self.enum_data_dicts, None)
# ================= 2. 执行量化 =================
def main(): input_model_path = "best.onnx" output_model_path = "quantized.onnx"
    # 直接根据你的截图，使用当前的 images 文件夹
    calib_dir = "images"
    
    # 获取 ONNX 模型的输入节点名称
    model = onnx.load(input_model_path) input_name = model.graph.input[0].name print(f"[INFO] 识别到模型输入节点名称为: {input_name}")
    # 【重要修改点】：请确保这个分辨率与你 YOLOv26nano 训练时的输入分辨率一致！ 如果你训练的是 640x640，保持不变；如果是 416x416，请修改这里。
    input_shape = (640, 640) print("[INFO] 正在初始化数据读取器...") data_reader = YOLODataReader(calib_dir, input_name, input_shape) print("[INFO] 
    开始进行量化，此过程可能需要几分钟...") vai_q_onnx.quantize_static(
        model_input=input_model_path, model_output=output_model_path, calibration_data_reader=data_reader, quant_format=vai_q_onnx.VitisQuantFormat.QDQ, 
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    ) print(f"[SUCCESS] 量化完成！已生成文件: {output_model_path}") if __name__ == "__main__":
    main()
