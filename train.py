import time
import copy

from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer, save_images

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    if opt.validate_freq > 0:
        validate_opt = copy.deepcopy(opt)
        validate_opt.phase = 'val'
        validate_opt.serial_batches = True  # no shuffle
        val_data_loader = CreateDataLoader(validate_opt)
        val_dataset = val_data_loader.load_data()
        val_dataset_size = len(val_data_loader)
        print('#validation images = %d' % val_dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.train()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)

            if not model.is_train():
                continue

            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_steps, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                print("experiment name:", opt.name)
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        if opt.validate_freq > 0 and epoch % opt.validate_freq == 0:
            model.eval()
            validation_loss_B = 0.0
            validation_loss_C = 0.0
            b = 0
            for i, data in enumerate(val_dataset):
                model.set_input(data)
                real_in, fake_out_B, real_out_B, fake_out, real_out, val_loss_B, val_loss_C = model.validate()
                validation_loss_B += val_loss_B
                validation_loss_C += val_loss_C
                b += 1
                ABC_path = data['ABC_path']
                # print("ABC_path len", len(ABC_path))
                # last batch will be smaller than batch size
                for i in range(len(ABC_path)):
                    ABC_path_i = ABC_path[i]
                    file_name = ABC_path_i.split('/')[-1].split('.')[0]
                    real_out_i = real_out[i].unsqueeze(0)
                    fake_out_i = fake_out[i].unsqueeze(0)
                    real_out_B_i = real_out_B[i].unsqueeze(0)
                    fake_out_B_i = fake_out_B[i].unsqueeze(0)
                    images = [real_out_i, fake_out_i, real_out_B_i, fake_out_B_i]
                    names = ['real', 'fake', 'real_B', 'fake_B']

                    img_path = str(epoch) + '_' + file_name
                    save_images(images, names, img_path, opt=validate_opt, aspect_ratio=1.0,
                                width=validate_opt.fineSize)
            validation_loss_B /= b
            validation_loss_C /= b
            visualizer.print_val_losses(epoch, {'val_l1_B': validation_loss_B, 'val_l1': validation_loss_C})

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
