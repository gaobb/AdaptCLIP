"""Testing script for AdaptCLIP anomaly detection model."""

import argparse
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from dataset import Dataset, PromptDataset
from tools import Evaluator, get_logger, get_transform, setup_seed, visualizer


def prompt_association(image_memory, patch_memory, target_class_name):
    patch_level_num = len(patch_memory[target_class_name[0]])
    retrive_image = []
    retrive_patch = [[] for i in range(patch_level_num)]

    for class_name in target_class_name:
        retrive_image.append(image_memory[class_name])  # S*D
        for l in range(patch_level_num):
            retrive_patch[l].append(patch_memory[class_name][l]) #

    retrive_image = torch.stack(retrive_image)  # B*S*D
    for l in range(patch_level_num):
        retrive_patch[l] = torch.stack(retrive_patch[l])  # B*S*L*D
    return retrive_image, retrive_patch


def build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, features_list, DPAM_layer):
    """Build few-shot prompt memory."""
    # initialize_memory
    feats_scale_num = len(features_list)
    prompt_image_memory = {}
    prompt_patch_memory = {}

    image_temp = []
    patch_temp = [[] for i in range(feats_scale_num)]
    cls_names_temp = []
    view_ids_temp = []

    for idx, items in enumerate(tqdm(prompt_dataloader)):
        cls_name = items['cls_name']
        prompt_image = items['img'].to(device)  # B*s*c*h*w
        prompt_mask = items['img_mask'].to(device)
        view_id = items['view_id']

        with torch.no_grad():
            image_feat, patch_feat = model.encode_image(prompt_image, features_list, DPAM_layer = DPAM_layer)

        cls_names_temp.extend(cls_name)
        image_temp.append(image_feat)
        view_ids_temp.extend(view_id)

        for i in range(feats_scale_num):
            patch_temp[i].append(patch_feat[i])


    image_temp = torch.cat(image_temp, dim=0)
    for i in range(feats_scale_num):
        patch_temp[i] = torch.cat(patch_temp[i], dim=0)

    for obj in obj_list:
        if len(view_list) > 1:
            for view_id in view_list:
                indice = (np.array(cls_names_temp) == obj) & (np.array(view_ids_temp) == view_id)
                obj_name = obj + '_' + view_id

                prompt_image_memory[obj_name] = image_temp[indice]
                prompt_patch_memory[obj_name] = []

                for i in range(feats_scale_num):
                    prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])
        else:
            indice = (np.array(cls_names_temp) == obj)
            obj_name = obj

            prompt_image_memory[obj_name] = image_temp[indice]
            prompt_patch_memory[obj_name] = []

            for i in range(feats_scale_num):
                prompt_patch_memory[obj_name].append(patch_temp[i][[indice]])

    return prompt_image_memory, prompt_patch_memory


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.test_data_path
    save_path = args.save_path
    dataset_name = args.dataset
    batch_size = args.batch_size
    k_shots = args.k_shots
    seed = args.seed
    vl_reduction = args.vl_reduction
    pq_mid_dim = args.pq_mid_dim
    pq_context = args.pq_context
    eval_metrics =  args.eval_metrics
    mode = 'test'

    log_file = f'{dataset_name}_{seed}seed_{k_shots}shot_{mode}_log.txt'
    logger = get_logger(save_path, log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pretrained_model == 'ViT-L/14@336px':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer = 20)
        patch_size = 14
        input_dim = 768
        DPAM_layer = 20
    if args.pretrained_model == 'VITB16_PLUS_240':
        model, _ = adaptcliplib.load(args.pretrained_model, device=device)
        model.visual.DAPM_replace(DPAM_layer = 10)
        patch_size = 16
        input_dim = 640
        DPAM_layer = 10

    preprocess, target_transform = get_transform(image_size=args.image_size)
    if dataset_name in ['Real-IAD-Variety', 'RealIAD']:
        sample_level = True
        prompt_data = PromptDataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                                    dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, \
                                    seed=seed, class_name=args.class_name)
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                            dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, \
                            seed=seed, class_name=args.class_name)
    else:
        prompt_data = PromptDataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                                    dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed)
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=target_transform, \
                            dataset_name=dataset_name, k_shots=k_shots, save_dir=save_path, mode=mode, seed=seed)
        sample_level = False
    prompt_dataloader = torch.utils.data.DataLoader(prompt_data, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    obj_list = test_data.obj_list
    view_list = test_data.view_list

    # ====================== Init Adapters ======================
    textual_learner = TextualAdapter(model.to("cpu"), img_size, args.n_ctx)
    visual_learner = VisualAdapter(img_size, patch_size, input_dim=input_dim, reduction=vl_reduction)
    pq_learner = PQAdapter(img_size, patch_size, context=pq_context, input_dim=input_dim, mid_dim=pq_mid_dim, layers_num=len(features_list))


    logger.info('\n' + "loading model from: " + args.checkpoint_path)
    checkpoint_adapter = torch.load(args.checkpoint_path)
    textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
    visual_learner.load_state_dict(checkpoint_adapter["visual_learner"])
    pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])


    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)

    model.eval()
    textual_learner.eval()
    visual_learner.eval()
    pq_learner.eval()


    textual_learner_parameters = sum(p.numel() for p in textual_learner.parameters())
    visual_learner_parameters = sum(p.numel() for p in visual_learner.parameters())
    pq_learner_parameters = sum(p.numel() for p in pq_learner.parameters())

    learned_parameters = textual_learner_parameters + visual_learner_parameters + pq_learner_parameters
    fixed_parameters = sum(p.numel() for p in model.parameters())


    print(f"textual_learner params:{(textual_learner_parameters):.0f}",
          f"visual_learner params:{(visual_learner_parameters)/1e+6:.1f}M",
          f"pq_learner params:{(pq_learner_parameters)/1e+6:.1f}M",
          f"learned all parameters:{(learned_parameters)/1e+6:.1f}M",
          f"fixed params:{(fixed_parameters)/1e+6:.1f}M",
          f"all params:{(learned_parameters+fixed_parameters)/1e+6:.1f}M"
     )


    # ====================== Initialize Evaluation Metrics ======================
    cpu_eva = False
    if cpu_eva:
        evaluator = Evaluator('cpu', metrics=eval_metrics, sample_level=sample_level)
    else:
        evaluator = Evaluator(device, metrics=eval_metrics, sample_level=sample_level)

    # ======================Text Encoder forward ======================
    textual_learner.prepare_static_text_feature(model)
    static_text_features = textual_learner.static_text_features

    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()


    # ====================== Few-shot Prompt Memory ======================
    if k_shots > 0:
        prompt_image_memory, prompt_patch_memory = build_prompt_memory(model, prompt_dataloader, device, obj_list, view_list, args.features_list, DPAM_layer)


    # ====================== Visual and Learner forward ======================
    sample_ids, gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, query_paths = [], [], [], [], [], [], []
    # nums = 0
    # total_time = 0
    for idx, items in enumerate(tqdm(test_dataloader)):
        query_image = items['img'].to(device)
        current_batchsize = query_image.shape[0]
        query_path = items['img_path']

        cls_name = items['cls_name']
        cls_id = items['cls_id']
        sample_id = items['sample_id']

        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['img_mask'][:, 0]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        gt_mask = gt_mask.to(device)

        # torch.cuda.synchronize()
        # start_time = time.time()

        with torch.no_grad():
            query_feats, query_patch_feats = model.encode_image(query_image, args.features_list, DPAM_layer = DPAM_layer)

        if k_shots > 0:
            if len(view_list) > 1:
                target_cls_name = [cls_name + '_' + view_id for cls_name, view_id in zip(cls_name, items['view_id'])]
            else:
                target_cls_name = cls_name
            prompt_feats, prompt_patch_feats = prompt_association(prompt_image_memory, prompt_patch_memory, target_cls_name)

        # ====================== CLIP Baseline ======================
        '''
        global_logit, local_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, static_text_features)
        local_map = local_map[:, 1].detach()

        global_score = global_logit.softmax(-1)
        global_score = global_score[:, 1].detach()
        '''

        # ====================== visual_adapter ======================
        if args.visual_learner:
            global_vl_logit, local_vl_map = visual_learner(query_feats, query_patch_feats, static_text_features)
            local_vl_map = local_vl_map[:, 1].detach()

            global_vl_score = global_vl_logit.softmax(-1)
            global_vl_score = global_vl_score[:, 1].detach()

        # ====================== textual_adapter ======================
        if args.textual_learner:
            global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)
            local_tl_map = local_tl_map[:, 1].detach()

            global_tl_score = global_tl_logit.softmax(-1)
            global_tl_score = global_tl_score[:, 1].detach()

        # ====================== pq_adapter ======================
        if args.pq_learner and k_shots > 0:

            global_pq_logit, local_pq_map_list, align_score_list = pq_learner(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)

            local_pq_map_list = [x[:, 1].unsqueeze(1) for x in local_pq_map_list]
            local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
            align_score = fusion_fun(align_score_list, fusion_type = 'harmonic_mean')[:, 0]

            if isinstance(global_pq_logit, list):
                global_pq_score = [x.softmax(-1).unsqueeze(-1) for x in global_pq_logit]
                global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1).detach()
                global_pq_score = global_pq_score[:, 1].detach()
            else:
                global_pq_score = global_pq_logit.softmax(-1)
                global_pq_score = global_pq_score[:, 1].detach()

        if k_shots > 0:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map], fusion_type = args.fusion_type)
            pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type = 'harmonic_mean')
            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in pixel_anomaly_map.cpu()], dim = 0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, global_pq_score], fusion_type = args.fusion_type)
            image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type = "harmonic_mean")

        else:
            # get pixel level prediction
            pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map], fusion_type = args.fusion_type)

            pixel_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in pixel_anomaly_map.cpu()], dim = 0)
            pixel_anomaly_map = pixel_anomaly_map.to(device)

            # get image level prediction
            anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
            image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, anomaly_map_max], fusion_type = args.fusion_type)


        if dataset_name in ['Real-IAD-Variety', 'RealIAD', 'bmad-medical']:
            resize_mask = 256
            if resize_mask is not None:
                pixel_anomaly_map = F.interpolate(pixel_anomaly_map[:, None], size=(resize_mask, resize_mask), mode='bilinear', align_corners=False)
                pixel_anomaly_map = pixel_anomaly_map[:, 0]
                gt_mask = F.interpolate(gt_mask[:, None], size=(resize_mask, resize_mask), mode='nearest')
                gt_mask = gt_mask.bool().int()

        pixel_anomaly_map  = torch.nan_to_num(pixel_anomaly_map,  nan=0.0, posinf=0.0, neginf=0.0)
        image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)

        sample_ids.append(np.array(sample_id))
        cls_names.append(np.array(cls_name))
        query_paths.append(np.array(query_path))
        if cpu_eva:
            gt_masks.append(gt_mask.int().cpu())
            pr_masks.append(pixel_anomaly_map.cpu())

            gt_anomalys.append(gt_anomaly.int().cpu())
            pr_anomalys.append(image_anomaly_pred.cpu())
        else:
            gt_masks.append(gt_mask.int())
            pr_masks.append(pixel_anomaly_map)


            gt_anomalys.append(gt_anomaly.int())
            pr_anomalys.append(image_anomaly_pred)

    # ====================== Evaluation ======================
    results_eval = dict(sample_ids=sample_ids, gt_masks=gt_masks, pr_masks=pr_masks, cls_names=cls_names, gt_anomalys=gt_anomalys, pr_anomalys=pr_anomalys, query_paths=query_paths)
    results_eval = {k: np.concatenate(v, axis=0) if k in ['cls_names', 'query_paths', 'sample_ids']  else torch.cat(v, dim=0) for k, v in results_eval.items()}


    # save results
    msg = {}
    for idx, cls_name in enumerate(tqdm(obj_list)):
        metric_results = evaluator.run(results_eval, cls_name, logger)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
        msg['Name'].append('Avg') if avg_act else None

        for metric in eval_metrics:
            metric_result = metric_results[metric] * 100

            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)

            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                msg[metric].append(metric_result_avg)

    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.1f', numalign="center", stralign="center", )
    logger.info('\n' + tab)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaptCLIP", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--pretrained_model", type=str, default='ViT-L/14@336px', help="pre-trained model name")
    parser.add_argument("--checkpoint_path", type=str, default='./adaptclip_checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--k_shots", type=int, default=1, help="how many normal samples")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'], help='evaluation metrics')
    parser.add_argument("--fusion_type", type=str, default="average_mean", help='fusion type')
    parser.add_argument("--vl_reduction", type=int, default=4, help="the reduction number of visual learner")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="the number of the first hidden layer in pqadapter")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature")
    parser.add_argument("--class_name", type=str, help="class name for a special dataset, for example, bottle in MVTec")
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
