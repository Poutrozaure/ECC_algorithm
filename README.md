ECC Algorithm to find common patch area between 2 images and resize target patch to source patch size

Kudos to https://github.com/ceciliavision/zoom-learn-zoom
I reused __utils.py__ and __utils_align.py__ and my __ECC.py__ is an adapted version of __main_align_camera.py__ for my usage.

1. First step is to install the good packages

```
!pip install -r Requirement.txt
```

2. Second is to run algorithm
```
!python ECC.py --source_smartphone /content/image_camera --target_DSLR /content/image_DSLR --multi_patches
```

options are the following :

[--output]: output folder to save the patches

[--source_smartphone]: source is the template, the image we don't want to modify

[--target_DSLR]: target is file we want to modify to fit the source image

[--patch_width]: width value for source patch

[--patch_height]: height value for source patch

[--stride]: stride value for multipatch search

[--multi_patches]: enable the multipatch option

[--stride_limit]: this parameter is used to defined the exclusion area of searching pairs patches

[--verbose]: enable the comments for debug
