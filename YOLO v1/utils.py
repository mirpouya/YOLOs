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
        

