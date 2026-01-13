from architecture import VAE
import torch
import torchvision.transforms.v2 as transforms
import os



def file_count(lib):
    """Counts only files in the specified directory (non-recursive)."""
    count = 0
    with os.scandir(lib) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_vae = VAE(device=dev)

trained_vae_state = torch.load("vae_model.pth")
my_vae.load_state_dict(trained_vae_state)
my_vae.eval()

images = []

my_vae.test_generation_quality()


generate_condition = True
if generate_condition:
    for i in range(4):
        number_img_T = my_vae.generate()[0] # The tensor representing the image
        print(number_img_T.shape)

        transform_img = transforms.ToPILImage()

        number_img = transform_img(number_img_T)
        number_img.show()
        number_img.save(f"gen images/same_f_number{file_count('gen images')+1}.png")
        images.append(number_img_T)

    print(torch.var(torch.stack(images)))
