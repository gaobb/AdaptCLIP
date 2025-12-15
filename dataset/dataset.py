import json
import os
import random

import numpy as np
import tifffile as tiff
import torch
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError


def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'mvtec3d':
        obj_list = ['bagel', 'carrot',  'dowel',  'potato', 'rope', 'cable_gland',  'cookie',
                    'foam', 'peach', 'tire']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'xmz_cropped' or dataset_name == 'xmz':
        obj_list = ['cls1', 'cls2']
    elif dataset_name == 'Kvasir':
        obj_list = ['colon']
    elif dataset_name == 'medical':
        obj_list = ['brain',  'liver',  'retinal']
    elif dataset_name == 'medical-cls':
        obj_list = ['Br35H', 'BrainMRI', 'HeadCT']  # 'COVID19'
    elif dataset_name == 'medical-seg':
        obj_list = ['ClinicDB',  'ColonDB', 'Endo',  'ISIC', 'Kvasir']  # 'TN3K'
    elif dataset_name == 'RealIAD':
        obj_list = ["audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery",  \
                    "plastic_nut", "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock",  \
                    "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper"]

    elif dataset_name == 'Real-IAD-Variety':
        obj_list = ["2pin_block_plug", "3_adapter", "3pin_aviation_connector", "4_wire_stepping_motor", "D-sub-connector", "DVD_switch", "LED_indicator", \
                    "PLCC_socket", "VR_joystick", "access_card", "accurate_detection_switch", "aircraft_model_head", "angled_toggle_switch", "audio_jack_socket", \
                    "bag_buckle", "ball_pin", "balun_transformer", "battery", "battery_holder_connector", "battery_socket_connector", "bend_connector", "blade_switch", \
                    "blue_light_switch", "bluetooth_module", "boost_converter_module", "bread_model", "brooch_clasp_accessory", "button_battery_holder", "button_motor", \
                    "button_switch", "car_door_lock_switch", "ceramic_fuse", "ceramic_wave_filter", "charging_port", "chip-inductor", "circuit_breaker", \
                    "circular_aviation_connector", "common-mode-choke", "common-mode-filter", "connector", "connector_housing-female", "console_switch", \
                    "crimp_st_cable_mount_box", "dc_jack", "dc_power_connector", "detection_switch", "duckbill_circuit_breaker", "earphone_audio_unit", \
                    "effect_transistor", "electronic_watch_movement", "ethernet-connector", "ferrite_bead", "ffc_connector_plug", "flow_control_valve", \
                    "flower_copper_shape", "flower_velvet_fabric", "fork-crimp-terminal", "fuse_cover", "fuse_holder", "gear", "gear_motor", "green-ring-filter", \
                    "hairdryer_switch", "hall_effect_sensor", "headphone_jack-female", "headphone_jack_socket", "hex_plug", "humidity_sensor", "ingot_buckle", \
                    "insect_metal_parts", "inverter_connector", "jam_jar_model", "joystick_switch", "kfc_push-key_switch", "knob-cap", "laser_diode", "lattice_block_plug", \
                    "lego-pin-connector-plate", "lego-propeller", "lego-reel", "lego-technical-gear", "lego-turbine", "lighting_connector", "lilypad_led", "limit-switch", \
                    "lithium_battery_plug", "littel-fuse", "little_cow_model", "lock", "long-zipper", "meteor_hammer_arrowhead", "miniature_laser_module", \
                    "miniature_lifting_motor", "miniature_motor", "miniature_stepper_motor", "mobile_charging_connector", "model_steering_module", "monitor_socket", \
                    "motor_bracket", "motor_gear_reducer", "motor_plug", "mouse_socket", "multi_function_switch", "nylon_ball_head", "optical_fiber_outlet", \
                    "pencil_sharpener", "pinboard_connector", "pitch_connector", "pneumatic_elbow", "pot_core", "potentiometer", "power_bank_module", \
                    "power_inductor", "power_jack", "power_strip_socket", "pulse_transformer", "purple-clay-pot", "push_button_switch", "push_in_terminal", \
                    "recorder_switch", "rectangular-connector-accessories", "red_terminal", "retaining_ring", "rheostat", "rotary_position_sensor", \
                    "round_twist_switch", "self-lock_switch", "shelf_support_module", "side_press_switch", "side_release_buckle", "silicon_cell_sensor", "sim_card_reader", \
                    "single-pole_potentiometer", "single_switch", "slide_switch", "slipper- model", "small_leaf", "smd_receiver_module", "solderless_adapter", \
                    "spherical_airstone", "spring_antenna", "square_terminal", "steering_T-head", "steering_wheel", "suction-cup", "telephone_spring_switch", \
                    "tension_snap_release_clip", "thumbtack", "thyristor", "toy-tire", "traceless_hair_clip", "travel_switch", "travel_switch_green", \
                    "tubular_key_switch", "vacuum_switch", "vehicle_harness_conductor", "vertical-adjustable-resistor", "vibration_motor", \
                    "volume_potentiometer", "wireless_receiver_module"]

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id


class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, k_shots, save_dir, mode='train', seed=10, class_name=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.k_shots = k_shots
        self.mode = mode
        self.save_dir = save_dir

        meta_info_json = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_test_info = meta_info_json['test']
        meta_train_info = meta_info_json['train']

        self.prompt_data_all = meta_train_info

        if class_name is not None:
            self.cls_names = [class_name]
        else:
            self.cls_names = list(meta_test_info.keys())

        if dataset_name == 'Real-IAD-Variety':
            self.view_list = ['C01', 'C02', 'C03', 'C04', "C05"]
        elif dataset_name == 'RealIAD':
            self.view_list = ['C1', 'C2', 'C3', 'C4', "C5"]
        else:
            self.view_list = ['0']

        self.obj_list = self.cls_names
        self.class_name_map_class_id = {}
        for k, index in zip(self.cls_names, range(len(self.cls_names))):
            self.class_name_map_class_id[k] = index

        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(meta_test_info[cls_name])

        self.length = len(self.data_all)


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # query image
        data = self.data_all[index]

        img_path, mask_path, cls_name, specie_name, anomaly, view_id = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly'], data['view_id'] if 'view_id' in data else '0'
        if len(self.view_list) > 1:
            sample_id = img_path.split('/')[-3] +  img_path.split('/')[-2]
        else:
            sample_id = img_path.split('/')[-1].split('.')[0]

        try:
            img = Image.open(os.path.join(self.root, img_path))
        except UnidentifiedImageError:
            img = tiff.imread(os.path.join(self.root, img_path))
            img = Image.fromarray(img)

        if img.mode == 'L':
            # 如果是灰度图像，直接复制三份
            img = Image.merge("RGB", (img.copy(), img.copy(), img.copy()))

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        # normal image --> for few shot setting
        prompt_image_list = []
        if self.mode == 'train':
            prompt_data = random.choice(self.prompt_data_all[cls_name])
            prompt_data_path = prompt_data['img_path']
            prompt_image = Image.open(os.path.join(self.root, prompt_data_path))
            if prompt_image.mode == 'L':
                prompt_image = Image.merge("RGB", (prompt_image.copy(), prompt_image.copy(), prompt_image.copy()))
            prompt_image = self.transform(prompt_image)
            prompt_image_list.append(prompt_image)

        if len(prompt_image_list) > 0:
            prompt_image_list = torch.stack(prompt_image_list)

        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, \
                'view_id': view_id, 'sample_id': sample_id, 'prompt_img': prompt_image_list, \
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}


class PromptDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, k_shots, save_dir, mode='test', seed=10, class_name=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.k_shots = k_shots
        self.mode = mode
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        if dataset_name == 'Real-IAD-Variety':
            self.view_list = ['C01', 'C02', 'C03', 'C04', "C05"]
        elif dataset_name == 'RealIAD':
            self.view_list = ['C1', 'C2', 'C3', 'C4', "C5"]
        else:
            self.view_list = ['0']

        self.prompt_data_all = []

        meta_info_json = json.load(open(f'{self.root}/meta.json', 'r'))
        if meta_info_json['train']:
            meta_train_info = meta_info_json['train']
        else:
            meta_train_info = meta_info_json['test']

        if class_name is not None:
            self.cls_names = [class_name]
        else:
            self.cls_names = list(meta_train_info.keys())
        self.obj_list = self.cls_names

        self.class_name_map_class_id = {}
        for k, index in zip(self.cls_names, range(len(self.cls_names))):
            self.class_name_map_class_id[k] = index

        if self.k_shots > 0:
            fs_txt = f'{dataset_name}_{seed}seed_{str(self.k_shots)}shot_{mode}_prompts.txt'
            prompt_save_dir = os.path.join(save_dir, fs_txt)
            if len(self.view_list) > 1:
                for cls_name in self.cls_names:
                    data_tmp = meta_train_info[cls_name]
                    for view_id in self.view_list:
                        torch.manual_seed(seed)
                        data_view_tmp = [item for item in data_tmp if item['view_id'] == view_id]
                        indices = torch.randint(0, len(data_view_tmp), (self.k_shots,))
                        self.prompt_data_all.extend([data_view_tmp[i] for i in indices])

                        for i in range(len(indices)):
                            with open(prompt_save_dir, "a") as f:
                                f.write(data_view_tmp[indices[i]]['img_path'] + '\n')

            else:
                for cls_name in self.cls_names:
                    data_tmp = meta_train_info[cls_name]
                    #data_tmp = [item for item in data_tmp if item['anomaly'] == 1] # 排除OK
                    torch.manual_seed(seed)
                    indices = torch.randint(0, len(data_tmp), (self.k_shots,))
                    self.prompt_data_all.extend([data_tmp[i] for i in indices])

                    for i in range(len(indices)):
                        with open(prompt_save_dir, "a") as f:
                            f.write(data_tmp[indices[i]]['img_path'] + '\n')

        self.length = len(self.prompt_data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.prompt_data_all[index]
        if len(self.view_list) > 1:
            img_path, mask_path, cls_name, specie_name, anomaly, view_id = data['img_path'], data['mask_path'], data['cls_name'], \
                                                                data['specie_name'], data['anomaly'], data['view_id']
            sample_id = img_path.split('/')[-3] +  img_path.split('/')[-2]
        else:
            img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                                data['specie_name'], data['anomaly']
            view_id = '0'
            sample_id = img_path.split('/')[-1].split('.')[0]

        img = Image.open(os.path.join(self.root, img_path))
        if img.mode == 'L':
            img = Image.merge("RGB", (img.copy(), img.copy(), img.copy()))

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, \
                'view_id': view_id if len(self.view_list) > 1 else '0', \
                'sample_id': sample_id, \
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
