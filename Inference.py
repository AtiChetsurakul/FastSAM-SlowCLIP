import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image

try:
    import open_clip
except Exception as e:
    print(f'{e}', 'we use open-clip instead please search on google about |open-clip| for more detail')
    raise

model_path = ''
img_path = ''
imgsz = 1024
iou = 0.9

text_prompt = ''
conf = 0.4
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
randomcolor = True
withContours = False
retina = True


model = FastSAM(model_path)

input = Image.open(img_path)

input = input.convert("RGB")
everything_results = model(
    input,
    device=device,
    retina_masks=retina,
    imgsz=imgsz,
    conf=conf,
    iou=iou    
    )
bboxes = None
points = None
point_label = None
prompt_process = FastSAMPrompt(input, everything_results, device=device)

if text_prompt != None:
    ann = prompt_process.text_prompt(text=text_prompt)

else:
    ann = prompt_process.everything_prompt()





# if __name__ == "__main__":
#     # args = parse_args()
#     main()#args)
