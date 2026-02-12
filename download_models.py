import os
import torch
import subprocess


def download_zip_from_box_url(models_dir, url):
    file_basename = os.path.basename(url)

    subprocess.run(['wget', '-P', models_dir, url])
    subprocess.run(['unzip', '-o', os.path.join(models_dir, file_basename), '-d', models_dir])
    subprocess.run(['rm', os.path.join(models_dir, file_basename)])

if __name__ == '__main__':
    """
    Download all the models that we use for stuff
    """
    models_dir = '/home/striest/temp_models'

    torch_hub_dir = os.path.join(models_dir, 'torch_hub')
    x = input('Models dir = {}. Continue? [Y/n]'.format(models_dir))
    if x == 'n':
        exit(0)

    os.makedirs(models_dir, exist_ok=True)
    torch.hub.set_dir(torch_hub_dir)

    subprocess.run(['touch', os.path.join(models_dir, 'CATKIN_IGNORE')])

    # OUR STUFF
    physics_atv_visual_mapping_url = 'https://cmu.box.com/shared/static/cqj1sz7am8iqv13ce8n0fy9z2enl9277.zip'

    maxent_irl_url = 'https://cmu.box.com/shared/static/mbtdjyf69pij1uju6lbav7wuoy975ski.zip'

    frontier_estimation_url = 'https://cmu.box.com/shared/static/7insm7a4z125b2bcu3fe0cpiuer0odhc'

    download_zip_from_box_url(models_dir, physics_atv_visual_mapping_url)
    # download_zip_from_box_url(models_dir, maxent_irl_url)
    download_zip_from_box_url(models_dir, frontier_estimation_url)

    # RADIO
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-b', progress=True, skip_validation=True, source='github') #  force_reload=True
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2', progress=True, skip_validation=True, source='github') #  force_reload=True
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='e-radio_v2', progress=True, skip_validation=True, source='github') #  force_reload=True

    # Dinov2
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github')

    # SAM
    # os.makedirs(os.path.join(models_dir, 'segment_anything_models'), exist_ok=True)
    # sam_urls = [
    #     'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    #     'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    #     'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    # ]

    # for sam_url in sam_urls:
    #     subprocess.run(['wget', '-P', os.path.join(models_dir, 'segment_anything_models'), sam_url])