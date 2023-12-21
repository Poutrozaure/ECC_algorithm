### Libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import shutil
import argparse
import cv2
import numpy as np
import cv2,os,argparse
import numpy as np
from timeit import default_timer as timer
import utils as utils
import utils_align as utils_align
import glob
import re

### functions
def natural_key(string_):
    """A key function for natural sort order."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=str, default= "/content/image_output", help="output folder to save the patches")
  parser.add_argument("--source_smartphone", type=str, default= "/content/camera", help="source is the template, the image we don't want to modify")
  parser.add_argument("--target_DSLR", type=str, default= "/content/dslr", help="target is file we want to modify to fit the source image")
  parser.add_argument("--patch_width", type=int, default=512)
  parser.add_argument("--patch_height", type=int, default=512)
  parser.add_argument("--stride", type=int, default=512)
  parser.add_argument('--multi_patches', default=False, action='store_true', help='Bool type, enable the multipatch option')
  parser.add_argument('--verbose', default=False, action='store_true', help='Bool type, enable the comments')
  parser.add_argument("--stride_limit", type=float, default=0.2,help="this parameter will only take 80% of the image centered")

  args = parser.parse_args()

  folder = args.output
  out_f = os.path.join(folder, 'aligned')
  out_sum = os.path.join(folder, 'compare')
  temp_inp = os.path.join(folder, 'temp')

  if not os.path.exists(args.output):
      os.mkdir(args.output)

  if not os.path.exists(out_f):
      os.mkdir(out_f)

  if not os.path.exists(os.path.join(out_f,'source')):
      os.mkdir(os.path.join(out_f,'source'))

  if not os.path.exists(os.path.join(out_f,'target')):
      os.mkdir(os.path.join(out_f,'target'))

  if not os.path.exists(out_sum):
      os.mkdir(out_sum)

  if not os.path.exists(temp_inp):
      os.mkdir(temp_inp)

  ref_ind = 0
  MOTION_MODEL = "ECC"
  tform_txt = os.path.join(folder, 'tform.txt')

  allfiles_source=sorted(os.listdir(args.source_smartphone), key=natural_key)
  allfiles_source=[filename for filename in allfiles_source if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]

  allfiles_target=sorted(os.listdir(args.target_DSLR), key=natural_key)
  allfiles_target=[filename for filename in allfiles_target if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]

  if(args.verbose):
    print(allfiles_source)
    print(allfiles_target)

  w = 2*args.patch_width
  h = 2*args.patch_height

  index_photo = 0
  if(args.multi_patches == False):
    while(index_photo < len(allfiles_source)):
        count = 0
        #we empty the temps file
        files = [f for f in os.listdir(temp_inp)]
        files=[filename for filename in files if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
        for f in files:
                #print(f)
                os.remove(os.path.join(temp_inp,f))

        if(args.verbose):
          print(allfiles_source[index_photo])
        shutil.copyfile(os.path.join(args.source_smartphone,allfiles_source[index_photo]), os.path.join(temp_inp,f"s{allfiles_source[index_photo]}"))
        shutil.copyfile(os.path.join(args.target_DSLR,allfiles_source[index_photo]), os.path.join(temp_inp,f"t{allfiles_source[index_photo]}"))

        allfiles=[f for f in os.listdir(temp_inp)]
        imlist=[filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
        if(args.verbose):
          print(imlist)
        num_img = len(imlist)

        images = []
        image_ds = []

        for impath in sorted(imlist):
            img_rgb = cv2.imread(os.path.join(temp_inp, impath), -1)

            #### adding this part to avoid the FOV difference btw the 2 camera images
            center = img_rgb.shape
            x = center[1]/2 - w/2
            y = center[0]/2 - h/2
            img_rgb = img_rgb[int(y):int(y+h), int(x):int(x+w)]
            ###########################################################################
            if(args.verbose):
              print(os.path.join(folder, 'cropped', impath))
            img_rgb = utils.image_float(img_rgb)  # normalize to [0, 1]
            images.append(img_rgb)
            img_rgb_ds = cv2.resize(img_rgb, None, fx=1./(2**3), fy=1./(2**3),
                interpolation=cv2.INTER_CUBIC)
            image_ds.append(img_rgb_ds)

        # operate on downsampled images
        image_ds = image_ds[ref_ind:]

        #################### ALIGN ####################
        height, width = img_rgb.shape[0:2]
        corner = np.array([[0,0,width,width],[0,height,0,height],[1,1,1,1]])

        alg_start = timer()
        images_gray = utils.bgr_gray(image_ds)

        try:
          t, t_inv, valid_id = utils_align.align_ecc(image_ds, images_gray, 0, thre=0.3)
          alg_end = timer()

          images_t, t, t_inv = utils_align.apply_transform(images, t, t_inv, MOTION_MODEL, scale=2 ** 3)

          with open(tform_txt, 'w') as out:
              for i, t_i in enumerate(t):
                  out.write("%05d-%05d:"%(1, i+1) + '\n')
                  np.savetxt(out, t_i, fmt="%.4f")

          for i in range(num_img):
              corner_out = np.matmul(np.vstack([np.array(t_inv[i]),[0,0,1]]),corner)
              corner_out[0,:] = np.divide(corner_out[0,:],corner_out[2,:])
              corner_out[1,:] = np.divide(corner_out[1,:],corner_out[2,:])
              corner_out = corner_out[..., np.newaxis]
              if i == 0:
                  corner_t = corner_out
              else:
                  corner_t = np.append(corner_t,corner_out,2)

          #print("Valid IDs: ",valid_id)
          images_t = list(images_t[i] for i in valid_id)
          images = list(images[i] for i in valid_id)
          imlist = list(imlist[i] for i in valid_id)
          num_img = len(images_t)

          ################ CROP & COMPARE ################
          min_w = np.max(corner_t[0,[0,1],:])
          min_w = int(np.max(np.ceil(min_w),0))
          min_h = np.max(corner_t[1,[0,2],:])
          min_h = int(np.max(np.ceil(min_h),0))
          max_w = np.min(corner_t[0,[2,3],:])
          max_w = int(np.floor(max_w))
          max_h = np.min(corner_t[1,[1,3],:])
          max_h = int(np.floor(max_h))

          with open(tform_txt, 'a') as out:
              out.write("corner:" + '\n')
              out.write("%05d %05d %05d %05d"%(min_h, max_h, min_w, max_w))
          out.close()

          if ref_ind == 0:
              sum_img_t, sum_img = utils_align.sum_aligned_image(images_t, images)

              i = 0
              if len(imlist)>1:
                      index_photo +=1
                      for impath in sorted(imlist):
                          if(impath[0]=="s"):
                            save_path = os.path.join(folder, 'aligned/source')
                          else : save_path = os.path.join(folder, 'aligned/target')
                          w = 2*args.patch_width
                          h = 2*args.patch_height
                          img_t = images_t[i]
                          i += 1
                          print("write to: ",(folder + 'aligned/' + impath))
                          img_t_crop = img_t[min_h:max_h,min_w:max_w,:]

                          center = img_t_crop.shape
                          x = center[1]/2 - w/4
                          y = center[0]/2 - h/4
                          img_t_crop = img_t_crop[int(y):int(y+(h/2)), int(x):int(x+(w/2))]

                          wt, ht = img_t_crop.shape[:2]
                          cv2.imwrite(os.path.join(save_path, f"{impath[1:-4]}_cropped_sample{count}.jpg"), np.uint8(255.*img_t_crop))

              else :
                  w+=100
                  h+=100
                  if(args.verbose):
                    ("issue patch is too small with picture",impath, "increasing patch size checking")
                    print("new size",w,h)
        except:
            w+=100
            h+=100
            if(args.verbose):
              print("issue ecc don't converge with picture",impath,"increasing patch size checking" )
              print("new size",w,h)
  else:
    for index_photo in range(len(allfiles_source)):
        count = 0
        #we empty the temps file
        files = [f for f in os.listdir(temp_inp)]
        files=[filename for filename in files if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
        
        for f in files:
                os.remove(os.path.join(temp_inp,f))

        if(args.verbose):
          print(allfiles_source[index_photo])

        shutil.copyfile(os.path.join(args.source_smartphone,allfiles_source[index_photo]), os.path.join(temp_inp,f"s{allfiles_source[index_photo]}"))
        shutil.copyfile(os.path.join(args.target_DSLR,allfiles_source[index_photo]), os.path.join(temp_inp,f"t{allfiles_source[index_photo]}"))

        allfiles=sorted([f for f in os.listdir(temp_inp)])
        imlist=[filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
        imlist_origin=[filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
        if(args.verbose):
          print(imlist)
        num_img = len(imlist)

        images = []
        image_ds = []

        img_source_test = cv2.imread(os.path.join(temp_inp,imlist[0]))
        center_source = img_source_test.shape

        for yy in range(int(args.stride_limit*center_source[0]), center_source[0] - args.patch_height*2 -int(args.stride_limit*center_source[0]) + 1, args.stride):
          for xx in range(int(args.stride_limit*center_source[1]), center_source[1] - args.patch_width*2 -int(args.stride_limit*center_source[1])+ 1, args.stride):
            count+=1
            imlist=[filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]
            images = []
            image_ds = []
            for impath in sorted(imlist_origin):
                img_rgb = cv2.imread(os.path.join(temp_inp, impath), -1)
                center_actual = img_rgb.shape[0:2]
                oversize_h = int((center_actual[0]-center_source[0])/2)
                oversize_w = int((center_actual[1]-center_source[1])/2)
                #### adding this part to avoid the FOV difference btw the 2 camera images
                img_rgb =img_rgb[yy+oversize_h:yy+oversize_h + args.patch_height*2, xx+oversize_w:xx+oversize_w + args.patch_width*2, :]
                ###########################################################################
                if(args.verbose):
                  print(os.path.join(folder, 'cropped', impath))
                img_rgb = utils.image_float(img_rgb)  # normalize to [0, 1]
                images.append(img_rgb)
                img_rgb_ds = cv2.resize(img_rgb, None, fx=1./(2**3), fy=1./(2**3),
                    interpolation=cv2.INTER_CUBIC)
                image_ds.append(img_rgb_ds)
                #print(img_rgb_ds.shape)

            # operate on downsampled images
            image_ds = image_ds[ref_ind:]

            #################### ALIGN ####################
            height, width = img_rgb.shape[0:2]
            corner = np.array([[0,0,width,width],[0,height,0,height],[1,1,1,1]])
            alg_start = timer()
            images_gray = utils.bgr_gray(image_ds)

            try:
              t, t_inv, valid_id = utils_align.align_ecc(image_ds, images_gray, 0, thre=0.3)
              alg_end = timer()

              images_t, t, t_inv = utils_align.apply_transform(images, t, t_inv, MOTION_MODEL, scale=2 ** 3)

              with open(tform_txt, 'w') as out:
                  for i, t_i in enumerate(t):
                      out.write("%05d-%05d:"%(1, i+1) + '\n')
                      np.savetxt(out, t_i, fmt="%.4f")

              for i in range(num_img):
                  corner_out = np.matmul(np.vstack([np.array(t_inv[i]),[0,0,1]]),corner)
                  corner_out[0,:] = np.divide(corner_out[0,:],corner_out[2,:])
                  corner_out[1,:] = np.divide(corner_out[1,:],corner_out[2,:])
                  corner_out = corner_out[..., np.newaxis]
                  if i == 0:
                      corner_t = corner_out
                  else:
                      corner_t = np.append(corner_t,corner_out,2)

              #print("Valid IDs: ",valid_id)
              images_t = list(images_t[i] for i in valid_id)
              images = list(images[i] for i in valid_id)
              imlist = list(imlist[i] for i in valid_id)
              num_img = len(images_t)

              ################ CROP & COMPARE ################
              min_w = np.max(corner_t[0,[0,1],:])
              min_w = int(np.max(np.ceil(min_w),0))
              min_h = np.max(corner_t[1,[0,2],:])
              min_h = int(np.max(np.ceil(min_h),0))
              max_w = np.min(corner_t[0,[2,3],:])
              max_w = int(np.floor(max_w))
              max_h = np.min(corner_t[1,[1,3],:])
              max_h = int(np.floor(max_h))

              with open(tform_txt, 'a') as out:
                  out.write("corner:" + '\n')
                  out.write("%05d %05d %05d %05d"%(min_h, max_h, min_w, max_w))
              out.close()

              if ref_ind == 0:
                  sum_img_t, sum_img = utils_align.sum_aligned_image(images_t, images)

                  i = 0
                  if len(imlist)>1:
                    for impath in sorted(imlist):
                        if(impath[0]=="s"):
                          save_path = os.path.join(folder, 'aligned/source')
                        else : save_path = os.path.join(folder, 'aligned/target')
                        img_t = images_t[i]
                        i += 1
                        print("write to: ",(folder + 'aligned/' + f"{impath[1:-4]}_cropped_sample{count}.jpg"))
                        img_t_crop = img_t[min_h:max_h,min_w:max_w,:]

                        center = img_t_crop.shape
                        x = center[1]/2 - w/4
                        y = center[0]/2 - h/4
                        img_t_crop = img_t_crop[int(y):int(y+(h/2)), int(x):int(x+(w/2))]

                        wt, ht = img_t_crop.shape[:2]
                        if(wt == w/2):
                          cv2.imwrite(os.path.join(save_path, f"{impath[1:-4]}_cropped_sample{count}.jpg"), np.uint8(255.*img_t_crop))
            except:
              pass