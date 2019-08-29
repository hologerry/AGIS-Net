import os
from itertools import islice

from data import CreateDataLoader
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class = %s' % (opt.name, opt.phase, opt.name))

l1_loss_file = os.path.join(opt.results_dir, opt.phase, "l1_loss.txt")
cnt = 0
mean_l1_loss = 0.0

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    ABC_path = data['ABC_path'][0]
    file_name = ABC_path.split('/')[-1].split('.')[0]
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    real_in, fake_out_B, real_out_B, fake_out, real_out, l1_loss = model.test()
    mean_l1_loss += l1_loss.item()
    cnt += 1

    images = [real_out, fake_out]
    names = ['real', 'fake']

    img_path = file_name
    save_images(images, names, img_path, webpage=webpage, width=opt.fineSize)

webpage.save()
mean_l1_loss /= cnt
with open(l1_loss_file, "w") as f:
    f.write(str(mean_l1_loss))
