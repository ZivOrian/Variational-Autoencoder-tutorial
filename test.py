from architecture import VAE
import torch
import torchvision.transforms.v2 as transforms



dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_vae = VAE()

trained_vae_state = torch.load("vae_model.pth")
my_vae.load_state_dict(trained_vae_state)
my_vae.eval()


number_img_T = my_vae.generate(dev) # The tensor representing the image
print(number_img_T.shape)

number_img_T = torch.reshape(number_img_T,(28,28))
transform_img = transforms.ToPILImage()

number_img = transform_img(number_img_T)
number_img.show()
number_img.save("same_f_number.png")
