"""
General-purpose training script for image-to-image translation.

- Uses a proper DataLoader so inputs are 5D for 3D convs: [N, C, D, H, W]
- Avoids shadowing the `data` module by not using a variable named `data`
"""

import time
from options.train_options import TrainOptions
import data
from models import create_model
from util.visualizer import Visualizer
from torch.utils.data import DataLoader
import traceback

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # (Optional) debug hook preserved from your version
    if getattr(opt, "debug", False):
        print("DEBUG MODE ACTIVATED.")
        import pydevd_pycharm
        Host_IP_address = '143.248.31.79'
        print(f"For debug, listening to...{Host_IP_address}")
        pydevd_pycharm.settrace(Host_IP_address, port=5678,
                                stdoutToServer=True, stderrToServer=True)

    # ---- dataset + dataloader ----
    dataset_class = data.find_dataset_using_name(opt.dataset_mode)
    dataset = dataset_class(opt)

    # For giant volumes, batch_size should almost always be 1
    if opt.batch_size != 1:
        print(f"[warn] With large 3D volumes, set batch_size=1. "
              f"You passed batch_size={opt.batch_size}.")

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=0,         # keep 0 on UCR/GT clusters & Windows
        pin_memory=False
    )

    # ---- model / visualizer ----
    model = create_model(opt)      # creates model from opt.model
    model.setup(opt)               # load/print nets; create schedulers
    visualizer = Visualizer(opt)   # display/save images and plots

# ---- iteration & epoch init ----
# If we resume from a checkpoint like --epoch iter_12000,
# use 12000 as the starting iteration counter.
    start_epoch = getattr(opt, "epoch_count", 1)
    max_epoch = getattr(opt, "n_epochs", 100) + getattr(opt, "n_epochs_decay", 100)

    start_iter = 0
    if isinstance(opt.epoch, str) and opt.epoch.startswith("iter_"):
        try:
            start_iter = int(opt.epoch.split("_")[1])
            print(f"[info] Resuming from iteration {start_iter} based on epoch='{opt.epoch}'")
        except ValueError:
            print(f"[warn] Could not parse iteration from epoch='{opt.epoch}'. "
                f"Starting from iter 0.")
            start_iter = 0

    total_iters = start_iter


    visualizer.reset()
    visualizer.display_model_hyperparameters()
    print("Model hyperparameters documented on tensorboard.")

    did_sanity_check = False

    # ---- training with epoch loop ----
    try:
        for epoch in range(start_epoch, max_epoch + 1):
            iter_data_time = time.time()
            print(f"\n========== Epoch {epoch}/{max_epoch} ==========")

            for i, batch in enumerate(loader):
                iter_start_time = time.time()
                # time to load data (since previous iter ended)
                t_data = iter_start_time - iter_data_time

                total_iters += 1
                # keep the model's internal counter in sync
                if hasattr(model, "set_total_iters"):
                    model.set_total_iters(total_iters)

                # IMPORTANT: use the loader batch
                model.set_input(batch)
                model.optimize_parameters()

                # display images
                if total_iters % opt.display_freq == 0:
                    model.compute_visuals()
                    visualizer.display_current_results(
                        model.get_current_visuals(), total_iters
                    )

                # print losses
                if total_iters % opt.print_freq == 0:
                    print("----------------------------------")
                    print(f"exp name: {opt.name}, gpu_id:{opt.gpu_ids}, "
                          f"epoch:{epoch}, iter:{total_iters}")
                    print("----------------------------------")
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time)  # per-iter compute time
                    visualizer.print_current_losses(
                        epoch, total_iters, losses, t_comp, t_data
                    )
                    if getattr(opt, "display_id", 0) > 0:
                        visualizer.plot_current_losses(
                            total_iters, losses, is_epoch=False
                        )

                # save latest model/visuals/hist
                if total_iters % opt.save_latest_freq == 0:
                    print("----------------------------------")
                    print(f"saving the latest model (iteration {total_iters})")
                    save_suffix = (
                        f"iter_{total_iters}"
                        if getattr(opt, "save_by_iter", False)
                        else "latest"
                    )
                    model.save_networks(save_suffix)
                    print(f"saving the current histogram (iteration {total_iters})")
                    visualizer.display_current_histogram(
                        model.get_current_visuals(), total_iters
                    )
                    print(f"saving the current visuals (iteration {total_iters})")
                    visualizer.save_current_visuals(
                        model.get_current_visuals(), total_iters
                    )
                    print("----------------------------------")

                # time anchor for next iteration's t_data
                iter_data_time = time.time()

                # one-time sanity checks on the very first effective iter
                if not did_sanity_check:
                    did_sanity_check = True
                    try:
                        real = getattr(model, "real", None)
                        if real is not None:
                            print(f"[sanity] input real shape: {tuple(real.shape)} "
                                  "# expect [N,C,D,H,W]")
                        fake = getattr(model, "fake", None)
                        rec = getattr(model, "rec", None)
                        if fake is not None:
                            print(f"[sanity] fake shape: {tuple(fake.shape)}")
                        if rec is not None:
                            print(f"[sanity] rec  shape: {tuple(rec.shape)}")

                        def finite(t):
                            return (
                                t is not None
                                and hasattr(t, "isfinite")
                                and t.isfinite().all().item()
                            )

                        print(f"[sanity] finite(fake): {finite(fake)}  "
                              f"finite(rec): {finite(rec)}")

                        # Plane weights (if present)
                        for attr in [
                            "lambda_plane_target",
                            "lambda_plane_source",
                            "lambda_plane_ref",
                        ]:
                            if hasattr(model, attr):
                                print(f"[sanity] {attr}: {getattr(model, attr)}")
                    except Exception as e:
                        print("[sanity] check failed:", e)
                        traceback.print_exc()

            # ---- end of epoch: update LR exactly once ----
            model.update_learning_rate()

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    except Exception:
        print("\n[error] Exception occurred during training:")
        traceback.print_exc()
    finally:
        # Always try to save something useful
        try:
            save_suffix = (
                f"iter_{total_iters}"
                if getattr(opt, "save_by_iter", False)
                else "latest"
            )
            print(f"[finalize] Saving networks ({save_suffix}) before exit.")
            model.save_networks(save_suffix)
        except Exception:
            pass
