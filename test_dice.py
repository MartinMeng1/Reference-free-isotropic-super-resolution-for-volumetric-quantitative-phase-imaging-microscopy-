"""General-purpose test script for image-to-image translation with 3D dice assembly."""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.assemble_dice import Assemble_Dice
from util import util
from skimage import io
import data
from tqdm import tqdm
import numpy as np
from data.image_folder import make_dataset
from tifffile import imsave


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # Ensure gan_mode exists for axial_to_lateral_gan_athena in test mode
    if not hasattr(opt, "gan_mode"):
        opt.gan_mode = "lsgan"

    ## DEBUG FLAG
    if opt.debug:
        print("DEBUG MODE ACTIVATED.")
        import pydevd_pycharm

        Host_IP_address = '143.248.31.79'
        print("For debug, listening to...{}".format(Host_IP_address))
        pydevd_pycharm.settrace(Host_IP_address, port=5678, stdoutToServer=True, stderrToServer=True)
    ##

    # hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset_class = data.find_dataset_using_name(opt.dataset_mode)
    dataset_tolook_shape = dataset_class(opt)
    dataset_size_original = dataset_tolook_shape.size_original()
    dataset_size = dataset_tolook_shape.size()
    print("original dataset_shape: " + str(dataset_size_original))

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # create results dir
    if opt.data_name is None:
        web_dir = os.path.join(
            opt.results_dir,
            opt.name,
            '{}_{}'.format(opt.phase, opt.epoch)
        )
    else:
        web_dir = os.path.join(
            opt.results_dir,
            opt.data_name + '_by_' + opt.name,
            '{}_{}'.format(opt.phase, opt.epoch)
        )
    print("web_dir: " + str(web_dir))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)

    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    dice_assembly = Assemble_Dice(opt)

    print("whole Image size: {}".format(dice_assembly.image_size))
    print("Whole image step counts y,x,z: {}".format(
        (dice_assembly.y_steps, dice_assembly.x_steps, dice_assembly.z_steps)))
    print("Whole image step counts: {}".format(
        dice_assembly.y_steps * dice_assembly.x_steps * dice_assembly.z_steps))

    if opt.eval:
        model.eval()

    for i, data_i in enumerate(tqdm(dataset)):
        model.set_input(data_i)
        model.test()
        visuals = model.get_current_visuals()
        dice_assembly.addToStack(visuals)

    print("Inference Done. ")

    dice_assembly.assemble_all()
    print("Image volume re-assembled.")
    img_whole_dict = dice_assembly.getDict()
    print("re-merged fake image shape: {}".format(img_whole_dict['fake'].shape))
    if 'rec' in img_whole_dict:
        print("re-merged rec image shape:  {}".format(img_whole_dict['rec'].shape))

    webpage_wholeimg = html.HTML(
        web_dir,
        'Whole_img: Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch)
    )

    if opt.data_type == 'uint16':
        data_range = 2 ** 16 - 1
    elif opt.data_type == 'uint8':
        data_range = 2 ** 8 - 1

    # volumes from assembler
    fake_volume = img_whole_dict['fake']
    print("Output (fake) data type is: " + str(fake_volume.dtype))

    rec_volume = img_whole_dict.get('rec', None)
    if rec_volume is not None:
        print("Output (rec) data type is: " + str(rec_volume.dtype))

    # note: we are not assembling 'real' in Assemble_Dice, so don't expect it here

    # -------- save volumes --------
    if opt.save_volume:
        util.mkdir(web_dir + '/volumes')

        # save fake volume
        if opt.load_iter > 0:
            output_xy_vol_path = web_dir + '/volumes/output_fake_volume_xy-view_iter-' + str(opt.load_iter) + '.tif'
        else:
            output_xy_vol_path = web_dir + '/volumes/output_fake_volume_xy-view_epoch-' + str(opt.epoch) + '.tif'
        imsave(output_xy_vol_path, fake_volume)
        print("Output fake volume is saved as a tiff file. ")

        # save rec volume (if available)
        if rec_volume is not None:
            if opt.load_iter > 0:
                output_rec_vol_path = web_dir + '/volumes/output_rec_volume_xy-view_iter-' + str(opt.load_iter) + '.tif'
            else:
                output_rec_vol_path = web_dir + '/volumes/output_rec_volume_xy-view_epoch-' + str(opt.epoch) + '.tif'
            imsave(output_rec_vol_path, rec_volume)
            print("Output rec volume is saved as a tiff file. ")

    # -------- projections (optional) --------
    if opt.save_projections:
        fake_proj_xy = np.amax(fake_volume, axis=0)
        fake_proj_xz = np.amax(fake_volume[:, 800:1100, :], axis=1)
        fake_proj_yz = np.amax(fake_volume[:, :, 200:500], axis=2)

        util.mkdir(web_dir + '/projections')

        util.save_image(fake_proj_xy, web_dir + '/projections/fake_xy_proj_epoch-' + str(opt.epoch) + '.tif')
        util.save_image(fake_proj_xz, web_dir + '/projections/fake_xz_proj_epoch-' + str(opt.epoch) + '.tif')
        util.save_image(fake_proj_yz, web_dir + '/projections/fake_yz_proj_epoch-' + str(opt.epoch) + '.tif')

    # -------- slice saving (optional) --------
    if opt.save_slices:
        output_xy_path = web_dir + '/images/output_xy/output_xy_'
        output_yz_path = web_dir + '/images/output_yz/output_yz_'
        output_xz_path = web_dir + '/images/output_xz/output_xz_'

        util.mkdir(web_dir + '/images/output_xy')
        util.mkdir(web_dir + '/images/output_yz')
        util.mkdir(web_dir + '/images/output_xz')

        Z, Y, X = fake_volume.shape

        # YZ slices: iterate over X
        for i in tqdm(range(X)):
            util.save_image(fake_volume[:, :, i], output_yz_path + str(i) + '.tif')

        # XZ slices: iterate over Y
        for i in range(Y):
            util.save_image(fake_volume[:, i, :], output_xz_path + str(i) + '.tif')

        # XY slices: iterate over Z
        for i in tqdm(range(Z)):
            snapshot_xy = dice_assembly.getSnapshots(i, slice_axis=0)
            util.save_image(fake_volume[i, :, :], output_xy_path + str(i) + '.tif')

    # -------- metrics vs GT (optional) --------
    if opt.dataroot_gt is not None:
        GT_path = make_dataset(opt.dataroot_gt, 1)[0]
        gt_volume = io.imread(GT_path)

        print("Calculating PSNR for the whole image volume...")

        datarange = 2 ** 8 - 1

        fake_vol_for_metric = util.normalize(util.standardize(fake_volume), data_type=np.uint8)
        gt_volume = util.normalize(util.standardize(gt_volume), data_type=np.uint8)

        psnr_output_gt = util.get_psnr(fake_vol_for_metric, gt_volume, datarange)
        print("Metrics are calculated.")

        message = 'Experiment Name: ' + opt.name + '\n'
        message += '---------------------------------------------------------\n'
        message += '\nWhole_volume\n'
        message += '---------------------------------------------------------\n'
        message += 'Network Output vs. Groundtruth\n'
        message += '(psnr: %.4f) \n' % (psnr_output_gt)
        message += '---------------------------------------------------------'

        print(message)
        filename = os.path.join(web_dir, 'metrics.txt')

        with open(filename, "a") as metric_file:
            metric_file.write('%s\n' % message)

    print("----Test done----")
