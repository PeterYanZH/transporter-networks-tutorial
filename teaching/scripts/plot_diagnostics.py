# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Publication-style views of stored arrays. Display ROIs never affect inference."""
import argparse,json
from pathlib import Path
from figure_style import add_font_arguments, apply_style
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
add_font_arguments(parser)
args=parser.parse_args()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

root=Path(args.root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
a=np.load(root/'results/model_arrays.npz');m=json.loads((root/'results/model_metrics.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
from figure_style import apply_style
apply_style(root, font_family=args.font_family, font_dir=args.font_dir)
NAVY='#18344C';TEAL='#008B8B';AMBER='#E5A12A';GRAY='#607080'
cmap=LinearSegmentedColormap.from_list('response',['#F5F8F8','#C8E7E4','#59B8AD','#008B8B','#18344C'])
meta={'display_only':True,'relative_maps_reference':'full-array maximum; clipped at 1e-4 for display','conditioning':{},'sensitivity':{},'translation':{}}
def save(fig,name):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)
def relative(x):return np.log10(np.maximum(x/max(float(x.max()),1e-30),1e-4))
def setup(ax,roi,ticks=True):
    r0,r1,c0,c1=roi;ax.set_xlim(c0-.5,c1-.5);ax.set_ylim(r1-.5,r0-.5);ax.set_aspect('equal')
    if ticks:ax.set_xlabel('column c');ax.set_ylabel('row r');ax.tick_params(length=2)
    else:ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#D1DADF');spine.set_linewidth(.6)
def context(ax,roi,opacity=.25):
    rgb=a['image'][:,:,:3]/255.;gray=rgb@np.array([.299,.587,.114]);light=.96-opacity*(gray-gray.min())/max(gray.max()-gray.min(),1e-8)
    display=np.repeat(light[:,:,None],3,axis=2)
    red=(rgb[:,:,0]>.4)&(rgb[:,:,0]>1.6*rgb[:,:,1])&(rgb[:,:,0]>1.6*rgb[:,:,2])
    display[red]=rgb[red]
    ax.imshow(display,interpolation='nearest');setup(ax,roi)
def outline(ax,roi):
    rgb=a['image'][:,:,:3]/255.;gray=rgb@np.array([.299,.587,.114]);ax.contour(gray,levels=[.27,.45],colors=['#9DA9AD'],linewidths=.55,alpha=.65)
def response(ax,data,roi):
    im=ax.imshow(relative(data),cmap=cmap,vmin=-4,vmax=0,interpolation='nearest');setup(ax,roi);outline(ax,roi);return im
def scalebar(ax,roi):
    r0,r1,c0,c1=roi; x=c0+5;y=r1-6
    ax.plot([x,x+16],[y,y],color=NAVY,lw=2);ax.text(x+8,y-2,'50 mm',ha='center',va='bottom',fontsize=7,color=NAVY)
def peaks(ax,q,ideal=None):
    ax.scatter(q[1],q[0],s=55,facecolors='none',edgecolors='#E17C22',lw=1.2,zorder=5)
    if ideal is not None:ax.scatter(ideal[1],ideal[0],s=35,c=NAVY,marker='+',lw=1.1,zorder=6)

rollouts=np.load(root/'results/rollouts_arrays.npz')
if 'camera_rgb' in rollouts:
    fig,axes=plt.subplots(1,4,figsize=(7.2,2.1),layout='constrained')
    for j in range(3):
        image=rollouts['camera_rgb'][j];roi=rollouts['camera_rois'][j].copy();ax=axes[j]
        red=(image[:,:,0]>100)&(image[:,:,0]>1.6*image[:,:,1])&(image[:,:,0]>1.6*image[:,:,2])
        rr,cc=np.nonzero(red)
        roi=np.array([max(0,min(roi[0],rr.min()-12)),min(image.shape[0],max(roi[1],rr.max()+13)),max(0,min(roi[2],cc.min()-12)),min(image.shape[1],max(roi[3],cc.max()+13))])
        # Square viewports make the four panels comparable without stretching.
        rh,cw=roi[1]-roi[0],roi[3]-roi[2];side=min(max(rh,cw),min(image.shape[:2]));rm=(roi[0]+roi[1])/2;cm=(roi[2]+roi[3])/2
        r0=int(np.clip(round(rm-side/2),0,image.shape[0]-side));c0=int(np.clip(round(cm-side/2),0,image.shape[1]-side));roi=(r0,r0+side,c0,c0+side)
        ax.imshow(image,interpolation='nearest');setup(ax,roi,ticks=False);ax.set_title(f'({chr(97+j)}) RGB-D view {j+1}',loc='left',fontsize=9)
    roi=(0,160,0,160);ax=axes[3];context(ax,roi,opacity=.5);ax.set_title('(d) Fused RGB-H',loc='left',fontsize=9);ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('');ax.set_ylabel('');ax.set_facecolor('#f2f4f4')
    p=m['conditioning'][0]['pick_rc'];q=m['conditioning'][0]['predicted_place_rc_bin'];ax.scatter(p[1],p[0],c=AMBER,marker='x',s=25);peaks(ax,q);scalebar(ax,roi)
    save(fig,'model_observation')

# Same source / target viewports in all columns preserve relative peak shifts.
source=(82,176,2,112);target=(0,66,0,66)
fig=plt.figure(figsize=(7.2,5.2));gs=fig.add_gridspec(2,3,height_ratios=[1.05,1],hspace=.48,wspace=.30,left=.07,right=.98,bottom=.16,top=.94)
for j,rec in enumerate(m['conditioning']):
    p=rec['pick_rc'];q=rec['predicted_place_rc_bin'];ideal=rec['ideal_place_rc']
    ax=fig.add_subplot(gs[0,j]);context(ax,source,opacity=.4)
    ax.add_patch(Rectangle((p[1]-32,p[0]-32),64,64,fill=False,ec=AMBER,lw=1.1));ax.scatter(p[1],p[0],s=45,c=AMBER,marker='x',lw=1.7)
    ax.set_title(f'({chr(97+j)}) Grasp condition {j+1}',loc='left');ax.set_xlabel('column c');
    if j:ax.set_ylabel('')
    ax=fig.add_subplot(gs[1,j]);im=response(ax,a['conditional_maps'][j],target);peaks(ax,q,ideal)
    ax.set_title(f'({chr(100+j)}) Target response',loc='left');ax.text(.98,.04,f'{rec["position_error_mm"]+1e-10:.2f} mm',transform=ax.transAxes,ha='right',fontsize=8,color=NAVY,bbox={'fc':'white','ec':'none','alpha':.9,'pad':2})
    if j:ax.set_ylabel('')
    data=a['conditional_maps'][j];r0,r1,c0,c1=target
    meta['conditioning'][str(j+1)]={'source_roi_rc':source,'target_roi_rc':target,'max_angle_map_sum_fraction_in_target_roi':float(data[r0:r1,c0:c1].sum(dtype=np.float64)/data.sum(dtype=np.float64))}
cax=fig.add_axes([.23,.065,.57,.021]);cb=fig.colorbar(ScalarMappable(norm=Normalize(-4,0),cmap=cmap),cax=cax,orientation='horizontal',ticks=[-4,-3,-2,-1,0]);cb.set_label('log10(relative place response); full-map peak = 1',fontsize=8)
save(fig,'model_conditioned')

# Discrete angle candidates: no interpolation or smoothing between measured bins.
curves=a['angle_curves'];fig,axes=plt.subplots(3,2,figsize=(7.2,3.4),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
angles=np.arange(36)*10
for j,curve in enumerate(curves):
    normalized=curve/curve.max()
    for k,ax in enumerate(axes[j]):
        logp=np.log10(np.maximum(normalized,1e-6))
        ax.vlines(angles,-6,logp,color=TEAL,lw=1);ax.scatter(angles,logp,s=9,color=TEAL,zorder=3)
        ax.set_ylim(-6.3,.55);ax.set_yticks([-6,-3,0]);ax.grid(axis='y',alpha=.16)
        ax.set_xlim((-5,355) if k==0 else (115,185));ax.set_xticks([0,90,180,270,350] if k==0 else [120,140,160,180])
        peak=int(np.argmax(curve))*10
        ax.annotate(f'{peak} deg',xy=(peak,0),xytext=(peak+13,-1.2),fontsize=8,color=NAVY,arrowprops={'arrowstyle':'-','lw':.6,'color':GRAY})
        if j<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('rotation candidate (degrees)')
    axes[j,0].set_ylabel(f'Grasp {j+1}',fontsize=9)
fig.supylabel('log10(relative P)',fontsize=10)
axes[0,0].set_title('(a) All 36 discrete candidates',loc='left');axes[0,1].set_title('(b) Common detail: 120-180 deg',loc='left')
save(fig,'model_angles')

# Translation plots use one fixed ROI, while TV keeps its original full interior.
roi=(96,164,24,92)
fig=plt.figure(figsize=(7.2,4.2));gs=fig.add_gridspec(2,3,height_ratios=[1.3,.7],left=.065,right=.98,top=.90,bottom=.12,hspace=.58,wspace=.40)
original=a['pick'];moved=a['translation_maps'][1];expect=a['translation_expected'][1]
for k,data,title in [(0,original,'(a) Original pick response'),(1,moved,'(b) Input shifted by 1 row')]:
    ax=fig.add_subplot(gs[0,k]);im=response(ax,data,roi);ax.set_title(title,loc='left',fontsize=9)
    if k:ax.set_ylabel('')
    cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.025,ticks=[-4,-2,0]);cb.ax.tick_params(labelsize=7)
diff=np.abs(moved-expect);ax=fig.add_subplot(gs[0,2]);im=ax.imshow(diff,cmap='YlOrBr',vmin=0,vmax=diff.max(),interpolation='nearest');setup(ax,roi);ax.set_ylabel('');ax.set_title('(c) Absolute map difference',loc='left',fontsize=9)
cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.025);cb.formatter.set_powerlimits((0,0));cb.update_ticks();cb.ax.tick_params(labelsize=7)
ax=fig.add_subplot(gs[1,:]); vals=[rec['pick_total_variation_interior'] for rec in m['translation']];xs=np.arange(6)
ax.bar(xs,vals,color=[GRAY,TEAL,TEAL,NAVY,NAVY,NAVY],width=.52);ax.set_xticks(xs,[str(tuple(rec['shift_rc'])) for rec in m['translation']]);ax.set_xlabel('input shift (row, column), pixels');ax.set_ylabel('Interior TV');ax.set_ylim(0,.135);ax.set_yticks([0,.05,.10]);ax.grid(axis='y',alpha=.15);ax.set_title('(d) Distribution discrepancy over the original interior domain',loc='left',fontsize=9)
for x,v in zip(xs,vals):ax.text(x,v+.004,'0' if v==0 else (f'{v:.4f}' if v>1e-5 else f'{v:.1e}'),ha='center',fontsize=8,color=NAVY)
meta['translation']={'display_roi_rc':roi,'metric_roi_rc':[32,288,32,128],'difference_max':float(diff.max())}
save(fig,'model_translation')

# Common viewport for context and derivatives. Light background means <=1e-4.
roi=(0,182,0,115);p=m['conditioning'][0]['pick_rc'];q=m['conditioning'][0]['predicted_place_rc_bin']
fig,axes=plt.subplots(1,3,figsize=(7.2,4.0),layout='constrained')
ax=axes[0];context(ax,roi,opacity=.65);ax.scatter(p[1],p[0],c=AMBER,marker='x',s=45);peaks(ax,q);ax.add_patch(Rectangle((p[1]-32,p[0]-32),64,64,fill=False,ec=AMBER,lw=1.4));ax.set_title('(a) Context and query window',loc='left',fontsize=9)
for j,(key,title) in enumerate([('query_gradient','(b) Query feature sensitivity'),('key_gradient','(c) Key feature sensitivity')],1):
    data=a[key][32:-32,32:-32];ax=axes[j];im=response(ax,data,roi);ax.set_title(title,loc='left',fontsize=9);ax.set_ylabel('')
    if j==1:
        ax.add_patch(Rectangle((p[1]-32,p[0]-32),64,64,fill=False,ec=AMBER,lw=1.1));ax.text(.5,.035,'Outside crop:\nno input path',transform=ax.transAxes,ha='center',fontsize=8,color=GRAY)
    else:peaks(ax,q)
    r0,r1,c0,c1=roi;meta['sensitivity'][key]={'display_roi_rc':roi,'gradient_norm_mass_in_roi':float(data[r0:r1,c0:c1].sum(dtype=np.float64)/data.sum(dtype=np.float64))}
cb=fig.colorbar(ScalarMappable(norm=Normalize(-4,0),cmap=cmap),ax=axes,orientation='horizontal',fraction=.045,pad=.08,shrink=.7,ticks=[-4,-3,-2,-1,0]);cb.set_label('log10(relative gradient norm); each full-map peak = 1',fontsize=8)
save(fig,'model_sensitivity')

# Keep uncropped response maps for auditing the publication insets.
fig,axes=plt.subplots(1,5,figsize=(8.2,3.4),layout='constrained')
for j in range(3):axes[j].imshow(relative(a['conditional_maps'][j]),cmap=cmap,vmin=-4,vmax=0);axes[j].set_title(f'Condition {j+1}')
for j,key in enumerate(['query_gradient','key_gradient'],3):axes[j].imshow(relative(a[key][32:-32,32:-32]),cmap=cmap,vmin=-4,vmax=0);axes[j].set_title(key.replace('_','\n'))
for ax in axes:ax.set_xlabel('column');ax.set_ylabel('row')
save(fig,'model_uncropped_reference')
(root/'results/figure_display_metrics.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print('Replotted conditioned, discrete-angle, translation, and sensitivity panels.')
