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
        bboxes, iou_threshold, threshold, box_format="corners"
):
    """
    Parameters:
        bboxes (list): list of lists, containing all bounding boxes info
            [predicted class, prob_score, x1, y1,x2, y2]
        iou_threshold (float): threshold to determine predicted boxes are correct
        threshold (float): threshold to remove predicted bounding boxes (independent of iou)
        box_format (str): midpoint or corners

    Returns:
        list of boxes after performing non max suppression given a specific iou threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    # sorting bboxes according to highest probability score
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format = box_format,
            )  <  iou_threshold 
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
