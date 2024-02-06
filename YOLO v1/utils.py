import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    """
    Parameters:
        boxes_preds (tensor): predictions of bounding boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): correct bounding boxes coordinates (BATCH_SIZE, 4)
        box_format (str): midpoint -> (x, y, w, h), corners -> (x1, y1, x2, y2)

    Returns: 
        tensors: intersection over unions for all examples
    """
    
    if box_format == "corners":
        # boxes_preds dimensions: (N, 4), N: number of boxes, 4: x1, y1, x2, y2
        # boxes_labels dimensions: (N, 4)
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
                  
        

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3]/2   # x - w/2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4]/2   # y - h/2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]/2   # x + w/2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]/2   # y + h/2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3]/2   
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4]/2   
        box2_x1 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]/2   
        box2_y1 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]/2

    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



# Non max suppression function

def non_max_suppression(
        predictions,
        iou_threshold,
        prob_threshold,
        box_format="corners"
):
    # predictions: list of predicted bounding boxes -> [[...], [...], ...]
    # in each prediction we have [predicted_class_label, probability of the bounding box, x1, y1, x2, y2]

    assert type(predictions) == list
    # keeping boxes higher than prob_threshold
    bboxes = [box for box in predictions if box[1] > prob_threshold]
    boxes_after_nms = []

    # sort bounding boxes with reference to highest probability
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)