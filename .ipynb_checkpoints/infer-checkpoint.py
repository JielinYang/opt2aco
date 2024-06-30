
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

content_img = '/root/autodl-tmp/optic/0001.jpg'
style_img = '/root/autodl-tmp/000034.jpg'
style_transfer = pipeline(Tasks.image_style_transfer, model_id='damo/cv_aams_style-transfer_damo')
result = style_transfer(dict(content = content_img, style = style_img))
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
