import os
import cv2
import json
import torch
import numpy as np
from PIL import Image

# annots, save_path, predictions, datamodule
def visualize(images, targets, predictions, annots_path, save_path, batch_idx):
    image = images[0]
    bbox = targets[0]
    pred = predictions[0]

        # tensor -> numpy 변환 (GPU -> CPU -> numpy)
    if torch.is_tensor(image):
        image = image.cpu().permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
        image = (image * 255).clip(0,255).astype(np.uint8)
    
    with open(os.path.join(annots_path, "instances_val2017.json"), 'r') as f:
        val_annots = json.load(f)
    categories = val_annots['categories']
    
    # for i, pred in enumerate(predictions):
    #     if isinstance(pred, list):  # batch의 결과인 경우
    #         pred = pred[0]  # 첫 번째 이미지
    
    #     # 원본 이미지 로드
    #     img = Image.open(datamodule.pred_dataset[i])
    #     img_np = np.array(img)
        
    # image_to_show = image.cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    # image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환

    # bbox_to_show = bbox.cpu().numpy()
    # pred_to_show = pred.cpu().numpy()

    # colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8).astype(np.float32) / 255.0
    # mask_colored = np.zeros((*bbox_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)'

    # for class_id in range(21):
    #     mask_colored[bbox_to_show == class_id] = colors[class_id]
    # pred_colored = np.zeros((*pred_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)
    # for class_id in range(21):
    #     pred_colored[pred_to_show == class_id] = colors[class_id]

    # combined = np.hstack((image_to_show, mask_colored, pred_colored))
    # cv2.imshow('Image and Mask', combined)
    # cv2.waitKey(0)



    for box, category_id, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if score > 0.75:  # confidence threshold
            box = box.cpu().numpy()
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255),
                2   # 글씨 두께
            )

            category_name = None
            for n in categories:
                if n['id'] == category_id.item():  # tensor -> python int
                    category_name = n['name']
                    break
            if category_name is None:
                category_name = 'unknown'

            text = f"{category_name}({category_id}): {(score*100):.2f}"
            cv2.putText(image, text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,   # OpenCV 이미지 좌표계: (0,0)왼쪽 상단 => y좌표 방향 반대
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.imshow('Image and bbox', image)
    # cv2.waitKey(0)
    save_dir = os.path.join(save_path, 'visualize_output')
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_img = Image.fromarray(image)
    visualize_img.save(os.path.join(save_dir, f'visualize_{batch_idx}.png'))