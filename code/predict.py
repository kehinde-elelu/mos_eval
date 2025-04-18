import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import laion_clap
from torch.utils.data import DataLoader
from tqdm import tqdm
from mos_track1 import MosPredictor, MyDataset
import datetime 
from utils import *
from model.CRNNs import CRNN10, CRNN10_v2

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs('data/result', exist_ok=True)
model_number = 65

def systemID(wavID):
    return wavID.replace("audiomos2025-track1-","").split('_')[0]

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="../data/MusicEval-phase1", required=False, help='Path of musiceval dataset')
    parser.add_argument('--ckptdir', type=str, required=False, default=f'../track1_ckpt/EXPNAME/best_ckpt_{model_number}', help='your finetuned ckpt path')
    args = parser.parse_args()

    UPSTREAM_MODEL = 'CLAP-music'
    DATADIR = args.datadir
    finetuned_checkpoint = args.ckptdir

    outfile = 'answer_track1.txt'
    system_csv_path = os.path.join(DATADIR, 'system_mos/system_mos_phase1.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if UPSTREAM_MODEL == 'CLAP-music':
        SSL_OUT_DIM= 512 
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel= 'HTSAT-base')
        # net = MosPredictor(model, SSL_OUT_DIM).to(device)
        net = CRNN10_v2(model, SSL_OUT_DIM).to(device)
        net.eval()
    else:
        print('*** ERROR *** ' + UPSTREAM_MODEL + ' not supported.')
        exit()


    # load ckpt
    ckpt=torch.load(finetuned_checkpoint, map_location = device)
    net.load_state_dict(ckpt)

    wavdir = os.path.join(DATADIR, 'wav')
    test_list = os.path.join(DATADIR, 'sets/dev_mos_list.txt')
    test_set = MyDataset(wavdir, test_list) 
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=test_set.collate_fn)

    total_loss = 0.0
    predictions_overall = { }  # overall prediction scores
    predictions_textual = { }  # textual prediction scores
    criterion = nn.L1Loss()
    print('Starting prediction')

    for i, data in enumerate(tqdm(test_loader, ncols=100), 0):
        wav, label1, label2, filenames = data 
        wav = wav.squeeze(1)
        text=get_texts_from_filename(filenames)

        label1 = label1.unsqueeze(1).to(device)
        label2 = label2.unsqueeze(1).to(device)
        with torch.no_grad():
            output1, output2 = net(wav, text)
        loss1 = criterion(output1, label1)
        loss2 = criterion(output2, label2)
        loss = (loss1+loss2) / 2
        total_loss += loss.item()
        output1 = output1.cpu().detach().numpy()[0]
        output2 = output2.cpu().detach().numpy()[0]
        predictions_overall[filenames[0]] = output1[0]
        predictions_textual[filenames[0]] = output2[0]
    print("evaluation total loss:",total_loss)

    truth_overall = { }  # overall true scores
    truth_textual = { }  # textual true scores
    testf = open(test_list, 'r')
    for line in testf:
        parts = line.strip().split(',')
        wavID = parts[0]
        MOS1 = float(parts[1])
        MOS2 = float(parts[2])
        truth_overall[wavID] = MOS1
        truth_textual[wavID] = MOS2

    # calculate metrics
    sorted_wavIDs = sorted(predictions_overall.keys())  # wavID includes '.wav'
    truth_overall_list = []    # list of overall true MOS, sorted by uttID
    truth_textual_list = []    # list of textual true MOS, sorted by uttID
    prediction_overall_list = []    # list of overall prediction MOS, sorted by uttID
    prediction_textual_list = []    # list of textual prediction MOS, sorted by uttID
    for wavID in sorted_wavIDs:
        # overall
        t1 = truth_overall[wavID]
        p1 = predictions_overall[wavID]
        truth_overall_list.append(t1)
        prediction_overall_list.append(p1)
        # textual
        t2 = truth_textual[wavID]
        p2 = predictions_textual[wavID]
        truth_textual_list.append(t2)
        prediction_textual_list.append(p2)

    truth_overall_array = np.array(truth_overall_list)
    pred_overall_array = np.array(prediction_overall_list)
    truth_textual_array = np.array(truth_textual_list)
    pred_textual_array = np.array(prediction_textual_list)
    
    # Open a log file to write the results
    log_file = open(f'data/result/results_{current_time}.txt', 'w')

    # Replace print statements with log_file.write
    log_file.write(f"=========={model_number}===========\n")
    log_file.write("==========UTTERANCE===========\n")
    log_file.write("======OVERALL QUALITY=======\n")
    MSE1 = np.mean((truth_overall_array - pred_overall_array) ** 2)
    log_file.write('[UTTERANCE] Test error= %f\n' % MSE1)
    LCC1 = np.corrcoef(truth_overall_array, pred_overall_array)
    log_file.write('[UTTERANCE] Linear correlation coefficient= %f\n' % LCC1[0][1])
    SRCC1 = scipy.stats.spearmanr(truth_overall_array.T, pred_overall_array.T)
    log_file.write('[UTTERANCE] Spearman rank correlation coefficient= %f\n' % SRCC1[0])
    KTAU1 = scipy.stats.kendalltau(truth_overall_array, pred_overall_array)
    log_file.write('[UTTERANCE] Kendall Tau rank correlation coefficient= %f\n' % KTAU1[0])
    log_file.write("======TEXTUAL ALIGNMENT=======\n")
    MSE2 = np.mean((truth_textual_array - pred_textual_array) ** 2)
    log_file.write('[UTTERANCE] Test error= %f\n' % MSE2)
    LCC2 = np.corrcoef(truth_textual_array, pred_textual_array)
    log_file.write('[UTTERANCE] Linear correlation coefficient= %f\n' % LCC2[0][1])
    SRCC2 = scipy.stats.spearmanr(truth_textual_array.T, pred_textual_array.T)
    log_file.write('[UTTERANCE] Spearman rank correlation coefficient= %f\n' % SRCC2[0])
    KTAU2 = scipy.stats.kendalltau(truth_textual_array, pred_textual_array)
    log_file.write('[UTTERANCE] Kendall Tau rank correlation coefficient= %f\n' % KTAU2[0])
    
    ### SYSTEM
    true_sys_MOS_avg_overall = { }  # dict{sysname:true_avg_mos}
    true_sys_MOS_avg_textual = { }

    csv_file = open(system_csv_path, 'r')   
    for line in csv_file:
        parts = line.strip().split(',')
        sysID = parts[0]    # S002
        MOS1 = float(parts[1])
        MOS2 = float(parts[2])
        true_sys_MOS_avg_overall[sysID] = MOS1
        true_sys_MOS_avg_textual[sysID] = MOS2
    
    pred_sys_MOSes_overall = { }  # dict{sysname:[wav_mos1,wav_mos2,...]}
    pred_sys_MOSes_textual = { }
    for wavID in sorted_wavIDs:
        sysID = systemID(wavID)
        noop1 = pred_sys_MOSes_overall.setdefault(sysID, [ ])
        noop2 = pred_sys_MOSes_textual.setdefault(sysID, [ ])
        pred_sys_MOSes_overall[sysID].append(predictions_overall[wavID])
        pred_sys_MOSes_textual[sysID].append(predictions_textual[wavID])
    
    pred_sys_MOS_avg_overall = { }  # dict{sysname:pred_avg_mos}
    pred_sys_MOS_avg_textual = { }
    for k, v in pred_sys_MOSes_overall.items():
        avg_MOS1 = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg_overall[k] = avg_MOS1
    for k, v in pred_sys_MOSes_textual.items():
        avg_MOS2 = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg_textual[k] = avg_MOS2

    # dict-->list
    pred_sysIDs = sorted(pred_sys_MOS_avg_overall.keys())
    sys_truth_overall_list = [ ]
    sys_truth_textual_list = [ ]
    sys_pred_overall_list = [ ]
    sys_pred_textual_list = [ ]
    for sysID in pred_sysIDs:
        sys_truth_overall_list.append(true_sys_MOS_avg_overall[sysID])
        sys_truth_textual_list.append(true_sys_MOS_avg_textual[sysID])
        sys_pred_overall_list.append(pred_sys_MOS_avg_overall[sysID])
        sys_pred_textual_list.append(pred_sys_MOS_avg_textual[sysID])
    
    # list-->np.array
    sys_truth_overall_array = np.array(sys_truth_overall_list)
    sys_truth_textual_array = np.array(sys_truth_textual_list)
    sys_pred_overall_array = np.array(sys_pred_overall_list)
    sys_pred_textual_array = np.array(sys_pred_textual_list)

    # Replace all print statements with log_file.write
    log_file.write("==========SYSTEM===========\n")
    log_file.write("======OVERALL QUALITY=======\n")
    MSE1 = np.mean((sys_truth_overall_array - sys_pred_overall_array) ** 2)
    log_file.write('[SYSTEM] Test error= %f\n' % MSE1)
    LCC1 = np.corrcoef(sys_truth_overall_array, sys_pred_overall_array)
    log_file.write('[SYSTEM] Linear correlation coefficient= %f\n' % LCC1[0][1])
    SRCC1 = scipy.stats.spearmanr(sys_truth_overall_array.T, sys_pred_overall_array.T)
    log_file.write('[SYSTEM] Spearman rank correlation coefficient= %f\n' % SRCC1[0])
    KTAU1 = scipy.stats.kendalltau(sys_truth_overall_array, sys_pred_overall_array)
    log_file.write('[SYSTEM] Kendall Tau rank correlation coefficient= %f\n' % KTAU1[0])
    log_file.write("======TEXTUAL ALIGNMENT=======\n")
    MSE2 = np.mean((sys_truth_textual_array - sys_pred_textual_array) ** 2)
    log_file.write('[SYSTEM] Test error= %f\n' % MSE2)
    LCC2 = np.corrcoef(sys_truth_textual_array, sys_pred_textual_array)
    log_file.write('[SYSTEM] Linear correlation coefficient= %f\n' % LCC2[0][1])
    SRCC2 = scipy.stats.spearmanr(sys_truth_textual_array.T, sys_pred_textual_array.T)
    log_file.write('[SYSTEM] Spearman rank correlation coefficient= %f\n' % SRCC2[0])
    KTAU2 = scipy.stats.kendalltau(sys_truth_textual_array, sys_pred_textual_array)
    log_file.write('[SYSTEM] Kendall Tau rank correlation coefficient= %f\n' % KTAU2[0])

    # Close the log file
    log_file.close()
    
    # generate answer.txt: wavid,overall_score,textual_score
    ans = open(outfile, 'w')
    for k, v in predictions_overall.items():
        outl = k.split('.')[0] + ',' + str(v) + ',' + str(predictions_textual[k]) + '\n'
        ans.write(outl)
    ans.close()

if __name__ == '__main__':
    main()



# evaluation total loss: 228.50585389137268
# ==========UTTERANCE===========
# ======OVERALL QUALITY=======
# [UTTERANCE] Test error= 0.468818
# [UTTERANCE] Linear correlation coefficient= 0.712596
# [UTTERANCE] Spearman rank correlation coefficient= 0.715274
# [UTTERANCE] Kendall Tau rank correlation coefficient= 0.536642
# ======TEXTUAL ALIGNMENT=======
# [UTTERANCE] Test error= 0.550059
# [UTTERANCE] Linear correlation coefficient= 0.599341
# [UTTERANCE] Spearman rank correlation coefficient= 0.573925
# [UTTERANCE] Kendall Tau rank correlation coefficient= 0.420546
# ==========SYSTEM===========
# ======OVERALL QUALITY=======
# [SYSTEM] Test error= 0.240479
# [SYSTEM] Linear correlation coefficient= 0.806361
# [SYSTEM] Spearman rank correlation coefficient= 0.764532
# [SYSTEM] Kendall Tau rank correlation coefficient= 0.591133
# ======TEXTUAL ALIGNMENT=======
# [SYSTEM] Test error= 0.204943
# [SYSTEM] Linear correlation coefficient= 0.742363
# [SYSTEM] Spearman rank correlation coefficient= 0.626108
# [SYSTEM] Kendall Tau rank correlation coefficient= 0.463054