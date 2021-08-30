# -*- coding: utf-8 -*-
import paddle
from utils import timer

from data import cfg


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return paddle.concat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax



def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return paddle.concat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]
    max_xy = paddle.minimum(box_a[:, :, 2:].unsqueeze(2).expand((n, A, B, 2)),
                       box_b[:, :, 2:].unsqueeze(1).expand((n, A, B, 2)))
    min_xy = paddle.maximum(box_a[:, :, :2].unsqueeze(2).expand((n, A, B, 2)),
                       box_b[:, :, :2].unsqueeze(1).expand((n, A, B, 2)))
    return paddle.clip(max_xy - min_xy, min=0).prod(3)  # inter


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.shape[0], box_b.shape[0]]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a.unsqueeze(0) #[None, ...]
        box_b = box_b.unsqueeze(0)

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def elemwise_box_iou(box_a, box_b):
    """ Does the same as above but instead of pairwise, elementwise along the inner dimension. """
    max_xy = paddle.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = paddle.max(box_a[:, :2], box_b[:, :2])
    inter = paddle.clip((max_xy - min_xy), min=0)
    inter = inter[:, 0] * inter[:, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a + area_b - inter
    union = paddle.clip(union, min=0.1)

    # Return value is [n] for inputs [n, 4]
    return paddle.clip(inter / union, max=1)


def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].
    Wait I thought this was "box_utils", why am I putting this in here?
    """

    masks_a = masks_a.reshape((masks_a.shape[0], -1)) 
    masks_b = masks_b.reshape((masks_b.shape[0], -1)) 

    intersection = paddle.mm(masks_a, paddle.transpose(masks_b, [1, 0])) 
    area_a = masks_a.sum(axis=1).unsqueeze(1)
    area_b = masks_b.sum(axis=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a


def elemwise_mask_iou(masks_a, masks_b):
    """ Does the same as above but instead of pairwise, elementwise along the outer dimension. """
    masks_a = masks_a.reshape(-1, masks_a.shape[-1])
    masks_b = masks_b.reshape(-1, masks_b.shape[-1])

    intersection = (masks_a * masks_b).sum(axis=0)
    area_a = masks_a.sum(axis=0)
    area_b = masks_b.sum(axis=0)

    # Return value is [n] for inputs [h, w, n]
    return paddle.clip(intersection / paddle.clip(area_a + area_b - intersection, min=0.1), max=1)



def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).
    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.shape[0]
    num_gt     = gt.shape[0]

    gt_w = (gt[:, 2] - gt[:, 0]).unsqueeze(-1).expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1]).unsqueeze(-1).expand(num_gt, num_priors)

    gt_mat =     gt.unsqueeze(1).expand(num_gt, num_priors, 4)
    pr_mat = priors.unsqueeze(0).expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -paddle.sqrt( (diff ** 2).sum(axis=2) )




def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)
    
    # Size [num_objects, num_priors]
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)

    # Size [num_priors] best ground truth for each prior
    best_truth_overlap = overlaps.max(0)
    best_truth_idx = overlaps.argmax(0)

    # We want to ensure that each gt gets used at least once so that we don't
    # waste any training data. In order to do that, find the max overlap anchor
    # with each gt, and force that anchor to use that gt.
    for _ in range(overlaps.shape[0]):
        # Find j, the gt with the highest overlap with a prior
        # In effect, this will loop through overlaps.shape[0] in a "smart" order,
        # always choosing the highest overlap first.
        best_prior_overlap = overlaps.max(1)
        best_prior_idx = overlaps.argmax(1)
        j = best_prior_overlap.argmax()#[1]

        # Find i, the highest overlap anchor with this gt
        #print(j,best_prior_idx)
        i = best_prior_idx[j]

        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps[:, i] = -1
        # Set all other overlaps with j to be -1 so that this loop never uses j again
        overlaps[j, :] = -1

        # Overwrite i's score to be 2 so it doesn't get thresholded ever
        best_truth_overlap[i] = 2
        # Set the gt to be used for i to be j, overwriting whatever was there
        best_truth_idx[i] = j
    # The shape of Tensor in list must be [1]. But received its shape is [19248]
    matches = paddle.gather(truths,best_truth_idx,0) # truths[best_truth_idx]            # Shape: [num_priors,4]
    conf = paddle.gather(labels,best_truth_idx,0) + 1  # labels[best_truth_idx] + 1           # Shape: [num_priors]

    conf = paddle.where(best_truth_overlap < pos_thresh,
                                -paddle.ones_like(conf), conf)
    conf = paddle.where(best_truth_overlap < neg_thresh,
                                paddle.zeros_like(conf), conf)
    #conf[best_truth_overlap < pos_thresh] = -1  # label as neutral
    #conf[best_truth_overlap < neg_thresh] =  0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap = crowd_overlaps.max(1)
        best_crowd_idx = crowd_overlaps.argmax(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[paddle.logical_and(conf <= 0, best_crowd_overlap > cfg.crowd_iou_threshold)] = -1
        # conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx]  = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf   # [num_priors] top class label for each prior
    idx_t[idx]  = best_truth_idx # [num_priors] indices for lookup


def encode(matched, priors, use_yolo_regressors:bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)

        loc = paddle.concat((
            boxes[:, :2] - priors[:, :2],
            paddle.log(boxes[:, 2:] / priors[:, 2:])
        ), 1)
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = paddle.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = paddle.concat([g_cxcy, g_wh], 1)  # [num_priors,4]
        
    return loc


def decode(loc, priors, use_yolo_regressors:bool=False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf
        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = paddle.concat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * paddle.exp(loc[:, 2:])
        ), 1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        
        boxes = paddle.concat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * paddle.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    
    return boxes



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return paddle.log(paddle.sum(paddle.exp(x-x_max), 1)) + x_max



def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.
    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.astype("int64") #.long()
        _x2 = _x2.astype("int64")
    x1 = paddle.minimum(_x1, _x2)
    x2 = paddle.maximum(_x1, _x2)
    x1 = paddle.clip(x1-padding, min=0)
    x2 = paddle.clip(x2+padding, max=img_size)

    return x1, x2



def crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = paddle.arange(w, dtype=x1.dtype).reshape((1, -1, 1)).expand((h, w, n))
    cols = paddle.arange(h, dtype=x1.dtype).reshape((-1, 1, 1)).expand((h, w, n))
    
    masks_left  = rows >= x1.reshape((1, 1, -1))
    masks_right = rows <  x2.reshape((1, 1, -1))
    masks_up    = cols >= y1.reshape((1, 1, -1))
    masks_down  = cols <  y2.reshape((1, 1, -1))
    crop_mask = masks_left
    for t in [masks_right, masks_up, masks_down]:
        crop_mask = paddle.logical_and(crop_mask,t)
        
    #crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return paddle.multiply(masks,crop_mask.astype("float32")) #masks * crop_mask.astype("float") #.float()


def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.
    In effect, this does
        out[i, j] = src[i, idx[i, j]]
    
    Both src and idx should have the same size.
    """

    offs = paddle.arange(idx.shape[0]).unsqueeze(-1).expand_as(idx)
    idx  = idx + offs * idx.shape[1]

    return src.reshape((-1))[idx.reshape((-1))].reshape((idx.shape))
