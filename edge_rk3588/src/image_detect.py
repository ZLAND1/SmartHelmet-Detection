import os
import cv2
from waring import waring_alarm
from rknn_executor import RKNN_model_container
import numpy as np
from dataset_utils import COCO_test_helper



OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)

CLASSES = ("two_wheeler","helmet","without_helmet")


def filter_boxes(boxes, box_confidences, box_class_probs):

    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    x = np.array(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = y_exp / y_exp.sum(axis=2, keepdims=True)

    acc_metrix = np.arange(mc).astype(np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(axis=2)

    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    colors = [(0, 0, 255), (255, 0, 0), (255, 255, 255)]
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        color = colors[cl]
        cv2.rectangle(image, (top, left), (right, bottom), color, 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        #             (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, '{0}'.format(CLASSES[cl]),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)        

def setup_model(args):
    model_path = args.model_path
    platform = 'rknn'
    model = RKNN_model_container(args.model_path,args.npu_core)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('--model_path', default='../model/yolo11_best.rknn', type=str,  help='rknn model path')
#     parser.add_argument('--npu_core', default=0, type=int,  help='0,1,2')
#     parser.add_argument('--img_save', action='store_true', default=True, help='save the result')
#     parser.add_argument('--img_folder', type=str, default='../image', help='image folder path')
#     args = parser.parse_args()
def start(img_name,model):
    # model, platform = setup_model(args)

    # file_list = sorted(os.listdir(args.img_folder))
    # img_list = []
    # for path in file_list:
    #     if img_check(path):
    #         img_list.append(path)
    img_folder='../image'
    img_save=True
    co_helper = COCO_test_helper(enable_letter_box=True)

    # for i in range(len(img_list)):
    #     print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

    #     img_name = img_list[i]
    #     img_path = os.path.join(args.img_folder, img_name)
    #     if not os.path.exists(img_path):
    #         print("{} is not found", img_name)
    #         continue

    img_src = cv2.imread(img_name)
       
    img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    # 最终输入到模型进行推理的图片是经过一系列预处理操作后的 img
    outputs = model.run([img])
    boxes, classes, scores = post_process(outputs)
    if boxes is not None:
        for i in range(len(classes)):
            if classes[i] == 2:                       
                waring_alarm()
                print("有未带头盔") 
                break
    else:
        print("都佩戴头盔")

    if  img_save:
        print('\n\n图片: {}'.format(img_name))
        #img_p = img_src.copy()
        if boxes is not None:
            draw(img_src, co_helper.get_real_box(boxes), scores, classes)

        if img_save:
            if not os.path.exists('./result'):
                os.mkdir('./result')
            parts = img_name.split("/")
            result_path = os.path.join('./result', parts[-1])
            cv2.imwrite(result_path, img_src)
            print('推理结果保存在 {}'.format(result_path))
            return img_src

