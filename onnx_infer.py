import os
import cv2
import glob
import time
import random
import onnxruntime
import numpy as np

def collect_images(data_dir):
    images = [[cv2.imread(path, 1), os.path.basename(path), path] for path in sorted(glob.glob(os.path.join(data_dir, "*.jpg")))]
    return images

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def data_preprocess(img0, img_size):
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]
    print(img.shape)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    assert len(img.shape) == 4
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def compute_intersection(set_1,set_2):
    '''计算两个集合之间的交集
    :param set_1: a array of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :param set_2: a array of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :return intersection: a array of shape (n1,n2)
    '''
    lower_bounds=np.maximum(set_1[:,:2].reshape(-1,1,2),set_2[:,:2].reshape(1,-1,2)) # (n1,n2,2)
    upper_bounds=np.minimum(set_1[:,2:].reshape(-1,1,2),set_2[:,2:].reshape(1,-1,2)) # (n1,n2,2)

    intersection=np.clip(upper_bounds-lower_bounds,a_min=0,a_max=None)  # (n1,n2,2)
    return intersection[:,:,0]*intersection[:,:,1]

def compute_iou(set_1,set_2):
    '''计算两个集合之间的IOU
    :param set_1: a array of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :param set_2: a array of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    :return iou: a array of shape (n1,n2)
    '''
    intersection=compute_intersection(set_1,set_2) # (n1,n2)
    area_set_1=(set_1[:,2]-set_1[:,0])*(set_1[:,3]-set_1[:,1])   # (n1,)
    area_set_2 = (set_2[:, 2] - set_2[:,0]) * (set_2[:, 3] - set_2[:, 1]) # (n2,)

    union=area_set_1.reshape(-1,1)+area_set_2.reshape(1,-1)-intersection # (n1,n2)
    return intersection/union # (n1,n2)

def nms_numpy(boxes, scores, iou_thres):
    '''
    :param boxes: ndarray(n,4)
    :param scores: ndarray(n,)
    :param iou_thres: float
    :return: indices=ndarray(n)
    '''
    output=[]
    sorted_indices=list(np.argsort(scores))[::-1]
    while len(sorted_indices)!=0:
        best=sorted_indices.pop(0)
        output.append(best)
        if len(sorted_indices)==0:
            break
        bb_xyxy=[]
        for bb in sorted_indices:
            bb_xyxy.append(boxes[bb])
        iou=compute_iou(np.array(boxes[best]).reshape(1,-1),np.array(bb_xyxy))[0]
        n=len(sorted_indices)
        sorted_indices=[sorted_indices[i] for i in range(n) if iou[i]<=iou_thres]
    return np.array(output)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.2,agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is np.float16:
        prediction = prediction.astype(np.float()) # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        conf,j=x[:,5:],np.zeros((x.shape[0],1))
        x = np.concatenate((box, conf, j.astype(np.float)), 1)[conf.reshape(-1) > conf_thres]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_numpy(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return np.array(output)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def make_grid(nx=20, ny=20):
    xv,yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float)

def sigmoid(x):
    '''
    :param x: ndarray
    :return: ndarray
    '''
    return np.tanh(x*0.5)*0.5+0.5

input_size = 640
sess = onnxruntime.InferenceSession('best.onnx')
input_name = sess.get_inputs()[0].name
output_names = []
for i in range(len(sess.get_outputs())):
    print('output shape:', sess.get_outputs()[i].name)
    output_names.append(sess.get_outputs()[i].name)

output_name = sess.get_outputs()[0].name
print('input name:%s, output name:%s' % (input_name, output_name))
input_shape = sess.get_inputs()[0].shape
print('input_shape:', input_shape)

names = ['head']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

save_img = True
out = './output'
data_dir = "../../human_head_dataset/images/val"
images = collect_images(data_dir)
strides=[32.,16.,8.]
anchor_grid=np.array([[[28,30],[36,35],[45,47]],[[19,25],[23,20],[23,29]],[[11,14],[16,20],[16,16]]],
                     dtype=np.float).reshape((3,1,-1,1,1,2))

for i in range(0, len(images)):
    if i >= (len(images) - 1):
        print("[INFO] Over.")
        break
    im0, fname0, path = images[i]
    print(path, fname0, ' shape:', im0.shape)
    img = im0.copy()
    image_tensor = data_preprocess(im0, input_size)
    print("[INFO] image batch shape:", image_tensor.shape)
    s = time.time()
    preds = sess.run(output_names, {input_name: image_tensor})
    output=[]
    for i,pred in enumerate(preds):
        bs, _, ny, nx,_ = pred.shape
        grid=make_grid(nx,ny)
        pred=sigmoid(pred)
        pred[...,0:2]=(pred[...,0:2]*2.-0.5+grid)*strides[i]
        pred[...,2:4]=(pred[...,2:4]*2)**2*anchor_grid[i]
        output.append(pred)
    preds = np.concatenate([pred.reshape((1, -1, 6)) for pred in output], axis=1)
    # post processing (nms)
    # Process detections
    preds=non_max_suppression(preds)
    for i, det in enumerate(preds):  # detections per image
        p = path,
        s = ''
        save_path = os.path.join(out, fname0)
        s += '%gx%g ' % img.shape[:2]  # print string
        print(s)
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image_tensor[0].shape[1:], det[:, :4], im0.shape).round()
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            # Write results
            # img=image_tensor[0].transpose(1,2,0)*255
            # img=img[:,:,::-1]
            # img = np.ascontiguousarray(img)
            for *xyxy, conf, cls in det:
                if save_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            cv2.imwrite(save_path,im0)