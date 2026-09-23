"""Read-only scientific checks; writes only review/check_results.json."""
import json
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from vaughan.config import PipelineConfig
from vaughan.data.dataset import Normalizer, _coarsen_sample, integrate_ice
from vaughan.physics.rtm import AnalyticRTM
from vaughan.physics.scatter import ScatteringRTM
from vaughan.assimilation.guidance import JointLikelihood, Observations
from vaughan.assimilation.sampler import SamplerOutput
from vaughan.inference.run_milton import RetrievalEngine
from vaughan.models.unet_xattn import CrossAttentionUNet
from vaughan.models.score_net import ScoreUNet

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
cfg = PipelineConfig.small_debug()
d = cfg.data
stats = {'temp': {'mean': [250.] * 10, 'std': [5.] * 10},
         'precip': {'mean': [0.], 'std': [1.]},
         'iwp': {'mean': [0.], 'std': [1.]}}
norm = Normalizer(stats)
rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, 8)
out = {}
# Physical arithmetic mean differs from mean of log1p state.
rain = torch.tensor([[[[0., 9.], [0., 9.]]]])
state = norm.physical_to_state(torch.full((1,10,2,2),250.), rain)[0]
s = {k: torch.zeros(1,2,2) for k in ['ir','ir_raw','mw','mw_raw','mw_zen']}
s.update(state=state, ir_mask=torch.ones(1,2,2), mw_mask=torch.ones(1,2,2))
coarse = _coarsen_sample(s,2)
out['rain_coarsening'] = {'physical_mean': rain.mean().item(),
    'implemented_mean': norm.state_to_physical(coarse['state'][None])[1].item()}
# --no-rtm works by large errors; allsky clips them back to 40 K.
cfg.guidance.sigma_ir_K = cfg.guidance.sigma_mw_K = 1e6
cfg.guidance.allsky = True
lik = JointLikelihood(rtm,norm,d,cfg.guidance)
t = torch.full((1,10,64,64),250.)
p = torch.zeros(1,1,64,64)
ir,mw = rtm(t,p)
x = norm.physical_to_state(t,p)
obs = Observations(x,ir-1,torch.ones(1,1,64,64),mw-1,torch.ones(1,1,8,8))
wi,wm = lik._allsky_weights(t,ir,mw,obs,lik.ir_w,lik.mw_w,0,25)
out['no_rtm_allsky'] = {'configured_sigma_K':1e6,'actual_sigma_K':float(wm[0,0,0,0]**-.5)}
# Profile integration is inconsistent across operators.
scat = ScatteringRTM(d.levels_hpa,d.ir_channels,d.mw_channels,8)
q = torch.full((1,10,1,1),1e-4)
out['ice_integration'] = {'normalizer_iwp':float(integrate_ice(q,d.levels_hpa)),
    'scattering_iwp':float(scat.ice_path_per_level({'ciwc':q},q.shape).sum()),
    'ir_iwp':float((q.flatten()*torch.cat([torch.diff(rtm.lnp.exp()*100),torch.diff(rtm.lnp.exp()*100)[-1:]])).sum()/9.80665)}
# When ice is supplied, neither operator depends on the rain input.
pr = torch.ones_like(p,requires_grad=True)
irs,mws = rtm(t,pr,{'iwp':torch.full_like(p,.1)})
out['rain_dependence_with_ice'] = {'ir_requires_grad':irs.requires_grad,'mw_requires_grad':mws.requires_grad}
# Serialization with a one-member ensemble; physical ice mean versus transform(mean).
engine = RetrievalEngine(cfg,norm,CrossAttentionUNet(d,cfg.unet),ScoreUNet(d,cfg.score),rtm,device=torch.device('cpu'))
scene=xr.Dataset({'lat':(('y','x'),np.zeros((64,64))), 'lon':(('y','x'),np.zeros((64,64)))})
samples=x[None]
result=engine.to_dataset(SamplerOutput(samples,x,torch.zeros_like(x),[]),obs,scene)
out['single_member_spread'] = {'temperature_all_nan':bool(result.temperature_spread.isnull().all()),
                             'precip_all_nan':bool(result.precip_spread.isnull().all())}
imem=torch.tensor([0.,np.log(10.)],dtype=torch.float32)
out['ice_ensemble_transform_example']={'physical_mean':float(torch.expm1(imem).mean()),
                                     'exported_transform_of_mean':float(torch.expm1(imem.mean()))}
# Inspect checkpoint metadata safely; never execute pickled code.
out['checkpoints']={}
for path in [ROOT/'unet.pt',ROOT/'score.pt',ROOT/'artifacts/checkpoints/unet.pt',ROOT/'artifacts/checkpoints/score.pt']:
    ck=torch.load(path,map_location='cpu',weights_only=True)
    out['checkpoints'][str(path.relative_to(ROOT))]={'step':ck.get('step'),'extra':ck.get('extra'),
        'keys':list(ck),'parameters_in_model':sum(v.numel() for v in ck['model'].values())}
    del ck
# Inspect cached real scenes, including gaps and overpass offsets.
out['real_scenes']=[]
for path in sorted((ROOT/'data/melissa/scenes').glob('*.nc')):
    with xr.open_dataset(path,engine='h5netcdf') as s:
        out['real_scenes'].append({'file':path.name,'shape':dict(s.sizes),'attrs':dict(s.attrs),
          'temp_nan':int(np.isnan(s.temp.values).sum()),'rain_nan':int(np.isnan(s.precip.values).sum()),
          'mw_coverage':float(s.mw_mask.mean()),'ir_coverage':float(s.ir_mask.mean())})
path=ROOT/'review/check_results.json'
path.write_text(json.dumps(out,indent=2,default=str))
print(json.dumps({k:v for k,v in out.items() if k!='real_scenes'},indent=2))
print('Real scenes',len(out['real_scenes']), 'saved',path)
