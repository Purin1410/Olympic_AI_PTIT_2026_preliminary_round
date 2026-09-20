"""Per-sample deterministic augmentation, pair-safe sampling and view controls."""
import io
import math
import random
import hashlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from .config import Config

MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)

def sample_seed(seed,epoch,pair_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{pair_id}'.encode()).digest()[:8],'little')

class Pairs(Dataset):
    def __init__(self,frame,root,config,training=False,epoch=0,norm=None,stress='native'):
        self.frame=frame.reset_index(drop=True);self.root=root;self.c=config
        self.training=training;self.epoch=epoch;self.norm=norm;self.stress=stress
        self.real=[];self.fake=[]
        if training and config.repair:
            for r in self.frame.itertuples():
                paths=[r.image_0,r.image_1];self.fake.append(paths[int(r.fake_position)]);self.real.append(paths[1-int(r.fake_position)])
    def __len__(self):return len(self.frame)
    def raw_views(self,path,rng):
        with Image.open(path) as original:im=original.convert('RGB')
        c=self.c;w,h=im.size
        if self.stress in ('jpeg95','jpeg85') or (self.training and c.augmentation=='jpeg' and rng.random()<.2):
            q=int(self.stress[-2:]) if self.stress.startswith('jpeg') else rng.randint(90,100)
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=q);buf.seek(0)
            with Image.open(buf) as rec:im=rec.convert('RGB')
        if self.stress=='resize' or (self.training and c.augmentation=='resize' and rng.random()<.2):
            factor=.75 if self.stress=='resize' else rng.uniform(.9,1.)
            im=TF.resize(TF.resize(im,[round(h*factor),round(w*factor)],antialias=True),[h,w],antialias=True)
        if self.stress=='blur':im=TF.gaussian_blur(im,[5,5],[.5,.5])
        if c.view.startswith('center'):
            frac=float(c.view[6:])/100;cw,ch=int(w*frac),int(h*frac)
            dx=round(rng.uniform(-.02,.02)*w) if self.training else 0
            dy=round(rng.uniform(-.02,.02)*h) if self.training else 0
            left=max(0,min(w-cw,(w-cw)//2+dx));top=max(0,min(h-ch,(h-ch)//2+dy))
            views=[TF.resize(im.crop((left,top,left+cw,top+ch)),[c.size,c.size],antialias=True)]
        else:
            centers=[(.5,.5)] if c.view=='single_native' else [(.5,.5),(.38,.38),(.62,.38),(.5,.64)]
            views=[]
            for cx,cy in centers:
                left=max(0,min(w-224,round(cx*w)-112));top=max(0,min(h-224,round(cy*h)-112))
                patch=im.crop((left,top,left+224,top+224))
                if c.view=='resampled':patch=TF.resize(TF.resize(patch,[112,112],antialias=True),[224,224],antialias=True)
                views.append(patch)
        flip=self.stress=='flip' or (self.training and rng.random()<.5)
        tensor=torch.stack([TF.to_tensor(TF.hflip(v) if flip else v) for v in views])
        if c.transform=='gaussian':tensor=tensor-TF.gaussian_blur(tensor,[5,5],[1.,1.])
        elif c.transform=='npr':tensor=tensor-torch.nn.functional.interpolate(torch.nn.functional.interpolate(tensor,scale_factor=.5,mode='nearest'),size=tensor.shape[-2:],mode='nearest')
        elif c.transform!='rgb':raise ValueError(c.transform)
        return tensor
    def __getitem__(self,index):
        row=self.frame.iloc[index];rng=random.Random(sample_seed(self.c.seed,self.epoch,row.pair_id))
        paths=[row.image_0,row.image_1];y=float(row.fake_position) if 'fake_position' in row else -1.
        if self.training and self.c.repair:
            y=float(rng.randrange(2));fake=rng.choice(self.fake);real=self.real[index]
            paths=[fake,real] if y==0 else [real,fake]
        # Same augmentation draw distribution and severity for both pair positions.
        pair_rng=rng.getstate();images=[]
        for path in paths:
            rng.setstate(pair_rng);x=self.raw_views(self.root/path,rng)
            if self.norm:
                mean=torch.tensor(self.norm['mean']).view(1,3,1,1);std=torch.tensor(self.norm['std']).view(1,3,1,1)
            else:mean,std=MEAN,STD
            images.append((x-mean)/std)
        return torch.stack(images),torch.tensor(y,dtype=torch.float32)

def make_loader(ds,shuffle=False,batch=None):
    gen=torch.Generator().manual_seed(sample_seed(ds.c.seed,ds.epoch,'order')%(2**63))
    return DataLoader(ds,batch_size=batch or ds.c.microbatch,shuffle=shuffle,
        num_workers=ds.c.workers if ds.training else 0,pin_memory=torch.cuda.is_available(),
        persistent_workers=False,generator=gen)

def fit_normalization(frame,root,c):
    ds=Pairs(frame,root,c);total=torch.zeros(3,dtype=torch.float64);sq=total.clone();n=0
    for row in frame.itertuples():
        for path in (row.image_0,row.image_1):
            x=ds.raw_views(root/path,random.Random(0)).double()
            total+=x.sum((0,2,3));sq+=(x*x).sum((0,2,3));n+=x.shape[0]*x.shape[2]*x.shape[3]
    mean=total/n;std=(sq/n-mean**2).clamp_min(1e-12).sqrt()
    return dict(mean=mean.tolist(),std=std.tolist(),images=2*len(frame),fit_pair_ids=frame.pair_id.tolist())
