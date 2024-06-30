
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os
import shutil

# content_img = '/root/autodl-tmp/optic/0001.jpg'
# style_img = '/root/autodl-tmp/000034.jpg'
# style_transfer = pipeline(Tasks.image_style_transfer, model_id='damo/cv_aams_style-transfer_damo')
# result = style_transfer(dict(content = content_img, style = style_img))
# cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])

opt_dir = '/root/autodl-tmp/optic'
aco_dir = '/root/autodl-tmp/style'
save_dir = '/root/autodl-tmp/acoustic'

def opt2aco(opt_dir, aco_dir, save_dir):
    # 获取光学、声学图像集合
    opt_files = os.listdir(opt_dir)
    aco_files = os.listdir(aco_dir)
    opt_files = [file for file in opt_files if os.path.isfile(os.path.join(opt_dir, file))]
    aco_files = [file for file in aco_files if os.path.isfile(os.path.join(aco_dir, file))]    
    # 准备模型文件
    style_transfer = pipeline(Tasks.image_style_transfer, model_id='damo/cv_aams_style-transfer_damo')
    for opt_file in opt_files:
        # 将光学图像复制到目标路径中
        opt_path = os.path.join(opt_dir, opt_file)
        save_path = os.path.join(save_dir, opt_file)
        shutil.copy(opt_path, save_path)
        # 使用多张风格图像进行转化
        for aco_file in aco_files:
            aco_path = os.path.join(aco_dir, aco_file)
            result = style_transfer(dict(content = save_path, style = aco_path))
            print(f"---use {aco_path} decorate {save_path}---")
        cv2.imwrite(save_path, result[OutputKeys.OUTPUT_IMG])
        print(f"---------{save_path}: style transfer finish!")


if __name__ == '__main__':
    opt2aco(opt_dir=opt_dir, aco_dir=aco_dir, save_dir=save_dir)