import torch
import open_clip


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b_b8k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')
# model, _, preprocess = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained='laion2b_s13b_b90k')
# tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')

tokenizer = open_clip.get_tokenizer('ViT-L-14')
