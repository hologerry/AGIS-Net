import os
import PIL.Image as Image

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

root_path = '/path/to/font_character_datasets'
mc_gan_datasets = 'mc-gan-datasets'

base_font = os.path.join(root_path, mc_gan_datasets, 'Capitals64/BASE/Code New Roman.0.0.png')
gray_path = os.path.join(root_path, mc_gan_datasets, 'public_web_fonts_gray_s_dataset')
texture_path = os.path.join(root_path, mc_gan_datasets, 'public_web_fonts_texture_dataset')

base_gray_texture_dataset_root = os.path.join(root_path, mc_gan_datasets, 'base_gray_texture_dataset')


gray_txt = 'public_web_fonts.txt'  # this file contains the font names

gray_txt_f = open(gray_txt, 'r')
gray_fonts = gray_txt_f.readlines()


if not os.path.exists(base_gray_texture_dataset_root):
    os.makedirs(base_gray_texture_dataset_root)

base_image = Image.open(base_font)

font_count = 0
train_count = 35  # unique, when train, copy val/* to train/
val_count = 0
test_count = 0


for idx, font in enumerate(gray_fonts):
    gray_font = os.path.join(gray_path, font[:-1])  # -1 is used to get rid of '\n'
    gray_image = Image.open(gray_font)

    texture_font = os.path.join(texture_path, font[:-1])
    texture_image = Image.open(texture_font)

    for char_id in range(26):
        pair = Image.new('RGB', (192, 64))
        char_base = base_image.crop((char_id*64, 0, char_id*64+64, 64))
        char_gray = gray_image.crop((char_id*64, 0, char_id*64+64, 64))
        char_texture = texture_image.crop((char_id*64, 0, char_id*64+64, 64))

        pair.paste(char_base)
        pair.paste(char_gray, (64, 0))
        pair.paste(char_texture, (128, 0))

        out_name = str(font_count+11000)

        pair_file = base_gray_texture_dataset_root + "/" + out_name + "_" + alphabets[char_id] + ".png"
        pair.save(pair_file)

    print("processed font: #", font_count)
    font_count += 1
