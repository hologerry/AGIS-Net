# Data preparation

## Convert font files to images
Please use the following code:
[font2image](https://github.com/hologerry/font2image)

## Prepocess the glyph images
Every glyph image is arranged as follow:

A: Base image, B: Shape image, C: Texture image

![](../imgs/11032_A.png)

code example: `generate_dataset_example.py`

## Prepare the pre-train dataset
```
xyz_dataset/
    train/
        XXX_XXX.png
        XXX_XXX.png
        ...
    val/
        XXX_XXX.png
        XXX_XXX.png
        ...
    test/
        XXX_XXX.png
        XXX_XXX.png
        ...
```

## Prepare the fine-tune datasets
```
abc_dataset/
    few_dict.txt        few-shot reference glyphs
    style/
        XXX_1.png
        XXX_2.png
        ...
    train/
        XXX_XXX.png
        XXX_XXX.png
        ...
    val/
        XXX_XXX.png
        XXX_XXX.png
        ...
```

Please read the `xxx_dataset.py` files, you can get more details.

## Chinese average font style(skeleton)
The average skeleton glyph image dataset including 6763 common used characters' glyph images with size 320X320. You can also download it via the `download_dataset.sh` script.