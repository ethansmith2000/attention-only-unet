import torch
import torchvision
from torchvision import transforms
from torch.optim import AdamW, Adam
import diffusers
from tqdm import tqdm
from models import UDiT_B, UDiT_L, UDiT_S


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 32
batch_size=32
lr = 0.4e-4
save_steps=500
# resume_ckpt = 'diffusion.ckpt'
resume_ckpt = None
gradient_checkpointing = False
epochs = 300


transform = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(size), 
                                transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


model = UDiT_S(input_size=size).to(device)
if gradient_checkpointing:
    model = model.enable_gradient_checkpointing()
scheduler = diffusers.DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="scheduler")

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-8)
nb_iter = 0

if resume_ckpt:
    state_dict = torch.load(resume_ckpt)
    nb_iter = state_dict['step']
    model.load_state_dict(state_dict['state_dict'])
    print('Loaded checkpoint')


@torch.no_grad()
def sample(num_steps=50, bs=128):
    xt = torch.randn(bs, 3, size, size).to(device)
    scheduler.set_timesteps(num_steps)
    for t in range(num_steps):
        pred = model(xt, t)
        xt = scheduler.step(pred, t, xt, return_dict=False)[0]

    return xt


print('Start training')
pbar = tqdm(total=epochs * len(dataloader), initial=nb_iter)
# torch.set_float32_matmul_precision("medium")
# model = torch.compile(model)
for current_epoch in range(100):
    for i, data in enumerate(dataloader):
        # with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
        x0 = (data[0].to(device)*2)-1
        noise = torch.randn_like(x0)
        t = torch.randint(0,scheduler.config.num_train_timesteps,(x0.shape[0],),device=device,).long()
        xt = scheduler.add_noise(x0, noise, t).to(x0.dtype)
        
        pred = model(xt, t)

        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        nb_iter += 1
        pbar.update(1)
        pbar.set_description(f'loss: {loss.item()}')

        if nb_iter % save_steps == 0:
            with torch.no_grad():
                print(f'Save export {nb_iter}')
                outputs = (sample(num_steps=128, bs=128) * 0.5) + 0.5
                torchvision.utils.save_image(outputs, f'export_{str(nb_iter).zfill(8)}.png')
                state_dict = {
                    "state_dict": model.state_dict(),
                    "step": nb_iter,
                }
                torch.save(state_dict, f'diffusion.ckpt')
