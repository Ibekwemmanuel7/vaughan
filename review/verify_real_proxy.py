import csv,json,time
from pathlib import Path
import torch,numpy as np
from vaughan.config import PipelineConfig
from vaughan.scripts.train import apply_preset
from vaughan.models.unet_xattn import CrossAttentionUNet
from vaughan.data.dataset import Normalizer,HurricaneSceneDataset,collate
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
cfg=PipelineConfig(); apply_preset(cfg,'small')
norm=Normalizer.load(str(ROOT/'norm_stats.json'))
net=CrossAttentionUNet(cfg.data,cfg.unet).eval()
net.load_state_dict(torch.load(ROOT/'unet.pt',map_location='cpu',weights_only=True)['model'])
paths=sorted((ROOT/'data/melissa/scenes').glob('*.nc'))
ds=HurricaneSceneDataset(paths,cfg.data,norm,downscale=2)
saved=list(csv.DictReader(open(ROOT/'results/melissa/melissa_scores.csv')))
rows=[]; start=time.perf_counter()
with torch.no_grad():
    for i,path in enumerate(paths):
        b=collate([ds[i]])
        x=net(b['ir'],b['ir_mask'],b['mw'],b['mw_mask'],b['mw_zen'])
        t,p=norm.state_to_physical(x); tt,pt=norm.state_to_physical(b['state'])
        c=t.shape[-1]//2; k=list(cfg.data.levels_hpa).index(300)
        sl=(0,k,slice(c-8,c+8),slice(c-8,c+8))
        err=float(((t[sl]-tt[sl])**2).mean().sqrt())
        ct,_=norm.state_to_physical(torch.zeros_like(x))
        rows.append({'scene':path.name,'core_rmse_300_K':err,
          'saved_core_rmse_300_unet_K':float(saved[i]['core_rmse_300_unet_K']),
          'climatology_core_rmse_300_K':float(((ct[sl]-tt[sl])**2).mean().sqrt()),
          'domain_rmse_300_K':float(((t[:,k]-tt[:,k])**2).mean().sqrt()),
          'rain_rmse_log_coarsened_label':float(((p-pt)**2).mean().sqrt())})
out={'elapsed_seconds':time.perf_counter()-start,'rows':rows,
     'mean_core_rmse':float(np.mean([r['core_rmse_300_K'] for r in rows])),
     'mean_climatology_core_rmse':float(np.mean([r['climatology_core_rmse_300_K'] for r in rows])),
     'max_difference_from_saved_core_metric':max(abs(r['core_rmse_300_K']-r['saved_core_rmse_300_unet_K']) for r in rows)}
(ROOT/'review/real_proxy_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
