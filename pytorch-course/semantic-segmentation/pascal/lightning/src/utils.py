import numpy as np
import cv2

SEED = 36


def visualize_batch(inputs, target, predictions):
    image, mask, pred = inputs[0], target[0], predictions[0]

    image_to_show = image.cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환

    mask_to_show = mask.cpu().numpy()
    pred_to_show = pred.cpu().numpy()
    colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8).astype(np.float32) / 255.0
    mask_colored = np.zeros((*mask_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)
    for class_id in range(21):
        mask_colored[mask_to_show == class_id] = colors[class_id]
    pred_colored = np.zeros((*pred_to_show.shape, 3), dtype=np.float32)  # (H, W, 3)
    for class_id in range(21):
        pred_colored[pred_to_show == class_id] = colors[class_id]

    combined = np.hstack((image_to_show, mask_colored, pred_colored))
    cv2.imshow('Image and Mask', combined)
    cv2.waitKey(0)
