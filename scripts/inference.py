import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import re
import cv2
import copy
import glob
import time
import gdown
import requests
import spaces
import torch
import pickle
import shutil
import json
import threading
import queue
import argparse
import gradio as gr
import numpy as np

from tqdm import tqdm
from argparse import Namespace
from huggingface_hub import snapshot_download
from moviepy.editor import *

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")


def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir + '/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir + '/dwpose',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")
        # gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
        file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        # resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")

        toc = time.time()
        print(f"download cost {toc - tic} seconds")

        for child in os.listdir(CheckpointsDir):
            child_path = os.path.join(CheckpointsDir, child)
            if os.path.isdir(child_path):
                print(child_path)
    else:
        print("Already download the model.")


download_model()  # for huggingface deployment.

from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending

Model_id = {""}

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            print("###### Start preprocess avatar data")
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list,
                                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            print("##### Feature extraction has been completed, Direct usage")
            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                print("#### bbox shift update! you need preprocess avatar data again")
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            # combine_frame = get_image(ori_frame,res_frame,bbox)
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            fps = 1 / (time.time() - start + 1e-6)
            print(f"Displaying the {self.idx}-th frame with FPS: {fps:.2f}")
            cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1

    def inference(self, audio_path, out_vid_name, fps):
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)

        # -------------------------------------------- extract audio feature -------------------------------------------
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"## Extract audio feature cost time:{(time.time() - start_time) * 1000}ms")

        # ------------------------------------------------ inference batch by batch ------------------------------------
        video_num = len(whisper_chunks)

        print("start inference")
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        start_time = time.time()
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        print(f"## processing audio cost time:{audio_path} costs {(time.time() - start_time) * 1000}ms")

        start_time = time.time()
        # res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(
                tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            # start_time = time.time()
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device)  # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)

            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()
        print(f"## batch inference costs time:{(time.time() - start_time) * 1000}ms")

        if out_vid_name is not None:
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, realtime, model_id=0):
    # 涉及到一个视频下载
    args_dict = {"result_dir": './results/output', "fps": 25, "batch_size": 8, "output_vid_name": "sn_test",
                 "use_saved_coord": False}  # same with inferenece script
    args = Namespace(**args_dict)

    total_start = time.time()

    if realtime:
        avatar = Avatar(avatar_id=model_id,
                        video_path=video_path,
                        bbox_shift=bbox_shift,
                        batch_size=args.batch_size,
                        preparation=True)
        avatar.inference(audio_path, args.output_vid_name, args.fps)

    else:

        # ------------------------------------- preprocess ----------------------------------------------------
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(args.result_dir, output_basename)  # related to video & audio inputs
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename + ".pkl")  # only related to video
        os.makedirs(result_img_save_path, exist_ok=True)

        if args.output_vid_name is None:
            output_vid_name = os.path.join(args.result_dir, output_basename + ".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)

        # ----------------------------------------- extract frames from source video -------------------------------
        extract_start = time.time()
        if get_file_type(video_path) == "video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full, exist_ok=True)
            try:
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            except:
                reader = imageio.get_reader(video_path)
                for i, im in enumerate(reader):
                    imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else:  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        print(f"##### Extarct frame from sorce video cost time:{time.time() - extract_start}s")

        # ---------------------------------------- extract audio feature ----------------------------------------------
        audio_start = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"##### Extract audio feature cost time:{time.time() - audio_start}s")

        # ---------------------------------------- extract input image landmark ---------------------------------------
        extract_landmark = time.time()
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path, 'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)  #
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
        # bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)  # 提前把这个指标设好，和get_landmark_and_bbox一致
        print(f"##### Extract image landmark cost time:{time.time() - extract_landmark}s")

        # ----------------------------------------- vae latents -----------------------------------------------------
        vae_start = time.time()
        # i = 0
        input_latent_list = []
        for bbox, frame in tqdm(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        print(f"##### VAE encode cost time:{time.time() - vae_start}s")

        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # ----------------------------------------------- inference batch by batch ------------------------------------
        unet_start = time.time()
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device)  # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)

            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
        print(f"##### Unet inference cost time:{time.time() - unet_start}s")

        # ---------------------------------------------pad to full image ---------------------------------------------
        print("pad talking image to original video")
        pad_start = time.time()
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % (len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue

            combine_frame = get_image(ori_frame, res_frame, bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
        print(f"##### Pad to full image cost time:{time.time() - pad_start}s")

        # --------------------------------------------- save video ---------------------------------------------------
        save_start = time.time()
        fps = 25
        output_video = "temp.mp4"
        images = []

        try:
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)
        except:
            def is_valid_image(file):
                pattern = re.compile(r'\d{8}\.png')
                return pattern.match(file)

            # images = []
            files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
            files.sort(key=lambda x: int(x.split('.')[0]))

            for file in files:
                filename = os.path.join(result_img_save_path, file)
                images.append(imageio.imread(filename))

            # 保存视频
            imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
        print(f"##### Save video cost time:{time.time() - save_start}s")

        # --------------------------------------------- compose audio -------------------------------------------------
        compose_start = time.time()
        input_video = './temp.mp4'
        # Check if the input_video and audio_path exist
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove("temp.mp4")
            shutil.rmtree(result_img_save_path)
            print(f"result is save to {output_vid_name}")
        except:
            # 读取视频
            reader = imageio.get_reader(input_video)
            fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

            # 将帧存储在列表中
            frames = images
            print(len(frames))

            # Load the video
            video_clip = VideoFileClip(input_video)

            # Load the audio
            audio_clip = AudioFileClip(audio_path)

            # Set the audio to the video
            video_clip = video_clip.set_audio(audio_clip)

            # Write the output video
            video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac', fps=25)

            os.remove("temp.mp4")
            # shutil.rmtree(result_img_save_path)
            print(f"result is save to {output_vid_name}")
        print(f"##### Compose video cost time:{time.time() - compose_start}s")
    print(f"###### Total cost time:{time.time() - total_start}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str,
                        default="/home/lgd/e_commerce_sd/app/sd_webui_musetalk/data/audio_0609.wav")
    parser.add_argument("--video_path", type=str, default="/home/lgd/e_commerce_sd/app/sd_webui_musetalk/data/g2.mp4")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--realtime", default=True)

    args = parser.parse_args()
    inference(args.audio_path, args.video_path, args.bbox_shift, args.realtime)
