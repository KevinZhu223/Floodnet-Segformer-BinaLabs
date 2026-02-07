import time
import json
import os
import shutil
import glob

# --- CONFIGURATION (Host Paths) ---
WATCH_DIR = '/media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/runs/floodnet_final_b4_ohem_cosine_V2'
SAFE_DIR = '/media/volume/Data_Kevin_Zhu/semseg_2d_code/semseg_2d/runs/floodnet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE'
# ---------------------

print(f"👀 Watching {WATCH_DIR} for FloodNet record-breaking models...")
if not os.path.exists(SAFE_DIR):
    os.makedirs(SAFE_DIR)
    print(f"📂 Created archive directory: {SAFE_DIR}")

best_acc_seen = 0.0
best_iou_seen = 0.0

while True:
    try:
        # 1. Find the latest checkpoint
        checkpoints = glob.glob(os.path.join(WATCH_DIR, 'checkpoint-*'))
        # Filter out our own backup folders to avoid recursion
        checkpoints = [c for c in checkpoints if "ARCHIVE" not in c and "SAVED" not in c]
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        if not checkpoints:
            time.sleep(30)
            continue

        latest_checkpoint = checkpoints[0]
        log_file = os.path.join(latest_checkpoint, 'trainer_state.json')
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            history = data.get('log_history', [])
            # SegFormer evaluation metrics names
            eval_entries = [x for x in history if 'eval_mAcc' in x and 'eval_mIoU' in x]
            
            if eval_entries:
                latest_entry = eval_entries[-1]
                curr_acc = latest_entry['eval_mAcc']
                curr_iou = latest_entry['eval_mIoU']
                curr_epoch = latest_entry['epoch']
                
                # --- CHECK 1: Is it a new ACCURACY Record? ---
                if curr_acc > best_acc_seen:
                    print(f"\n🚀 NEW ACCURACY RECORD! Epoch {curr_epoch} | Acc: {curr_acc:.5f}")
                    
                    safe_name = f"checkpoint-ACC-{curr_acc:.4f}-Ep{curr_epoch}"
                    dest_path = os.path.join(SAFE_DIR, safe_name)
                    
                    if not os.path.exists(dest_path):
                        print(f"   💾 Backing up {latest_checkpoint.split('/')[-1]} to {safe_name}...")
                        shutil.copytree(latest_checkpoint, dest_path)
                    
                    best_acc_seen = curr_acc

                # --- CHECK 2: Is it a new mIoU Record? ---
                if curr_iou > best_iou_seen:
                    print(f"\n🏆 NEW mIoU RECORD!      Epoch {curr_epoch} | mIoU: {curr_iou:.5f}")
                    
                    safe_name = f"checkpoint-mIoU-{curr_iou:.4f}-Ep{curr_epoch}"
                    dest_path = os.path.join(SAFE_DIR, safe_name)
                    
                    # Only copy if we didn't just copy it for Accuracy (avoid duplicates)
                    if not os.path.exists(dest_path):
                        # Check if this epoch was already saved under ACC
                        if curr_acc >= best_acc_seen:
                             # Already handled by the ACC check if it was the same checkpoint
                             pass 
                        else:
                            print(f"   💾 Backing up {latest_checkpoint.split('/')[-1]} to {safe_name}...")
                            shutil.copytree(latest_checkpoint, dest_path)
                    
                    best_iou_seen = curr_iou
                
                # Heartbeat Status
                print(f"\rBest Acc: {best_acc_seen:.4f} | Best mIoU: {best_iou_seen:.4f} | Current: {curr_acc:.4f} Acc / {curr_iou:.4f} mIoU", end="")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")

    time.sleep(60)
