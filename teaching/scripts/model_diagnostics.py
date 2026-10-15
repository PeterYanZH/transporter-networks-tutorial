# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Bounded, inference-only teaching diagnostics of the official Transporter."""
import argparse, json, time
from pathlib import Path
from figure_style import add_font_arguments, apply_style
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project containing upstream/ and checkpoints/')
parser.add_argument('--out', type=Path, default=Path('teaching'), help='Results and figures destination, relative to the current directory')
parser.add_argument('--agent-name', default='block-insertion-transporter-10-0')
parser.add_argument('--checkpoint', type=int, default=10000)
add_font_arguments(parser)
args=parser.parse_args()
project_root=args.project_root.expanduser().resolve()
assets=project_root/'upstream/ravens/environments/assets'
if not assets.is_dir():
    parser.error('Missing simulator assets below --project-root; run the repository setup first')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_erosion
import tensorflow as tf
import pybullet as pb
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.environments.environment import Environment
from ravens.tasks.block_insertion import BlockInsertion
from ravens.utils import utils

out=Path(args.out); figs=out/'figures'; results=out/'results'
figs.mkdir(parents=True,exist_ok=True); results.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
apply_style(font_family=args.font_family, font_dir=args.font_dir)
NAVY='#18344C'; TEAL='#008B8B'; AMBER='#E5A12A'
def save(fig,name):
    fig.savefig(figs/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(figs/f'{name}.png',dpi=240,bbox_inches='tight')
    plt.close(fig)
def panel(ax,img,title,cmap=None):
    ax.imshow(img,cmap=cmap,interpolation='nearest'); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
def heat(a):
    a=np.asarray(a); return np.log10(np.maximum(a/a.max(),1e-5))
def maps(agent,img,pick=None):
    attention=agent.attention.forward(img)
    p=tuple(np.unravel_index(attention.argmax(),attention.shape)[:2]) if pick is None else tuple(pick)
    transport=agent.transport.forward(img,p)
    q=np.unravel_index(transport.argmax(),transport.shape)
    return attention[:,:,0],transport,p,tuple(int(x) for x in q)
def action(agent,img,p,q):
    h=img[:,:,3]
    return {'pose0':(np.asarray(utils.pix_to_xyz(p,h,agent.bounds,agent.pix_size)),np.asarray(utils.eulerXYZ_to_quatXYZW((0,0,0)))),
            'pose1':(np.asarray(utils.pix_to_xyz(q[:2],h,agent.bounds,agent.pix_size)),np.asarray(utils.eulerXYZ_to_quatXYZW((0,0,-q[2]*2*np.pi/36))))}
def shift(a,dr,dc):
    b=np.zeros_like(a); h,w=a.shape[:2]
    b[max(dr,0):min(h,h+dr),max(dc,0):min(w,w+dc)]=a[max(-dr,0):min(h,h-dr),max(-dc,0):min(w,w-dc)]
    return b

np.random.seed(0); tf.random.set_seed(0); tf.keras.backend.set_learning_phase(0)
agent=OriginalTransporterAgent(args.agent_name,'block-insertion',str(project_root))
agent.load(args.checkpoint)
env=Environment(str(assets),disp=False,hz=480)
records=[]; views=[]; initial=None
started=time.time()
try:
    for seed in [1,3,5]:
        np.random.seed(seed); env.seed(seed); task=BlockInsertion(); task.mode='test'; env.set_task(task)
        obs=env.reset(); img=agent.get_image(obs)
        ap,tp,p,q=maps(agent,img)
        if seed==1:
            initial=(img.copy(),ap.copy(),tp.copy(),p,q)
            block=task.goals[0][0][0][0]; object_pose=pb.getBasePositionAndOrientation(block)
            target_pose=task.goals[0][2][0]
            obs_initial=obs
        total=0.; actions=[]
        for step in range(task.max_steps):
            if step:
                img_current=agent.get_image(obs); a0,t0,p0,q0=maps(agent,img_current)
            else: img_current,p0,q0=img,p,q
            act=action(agent,img_current,p0,q0)
            obs,reward,done,info=env.step(act); total+=reward
            actions.append({'pick_rc':list(map(int,p0)),'place_rc_rotation':list(q0),'reward':float(reward),'done':bool(done)})
            if done: break
        after=agent.get_image(obs)
        records.append({'seed':seed,'reward':float(total),'actions':actions})
        views.append((img,ap,tp.max(2),p,q,after))
        print('rollout',seed,'reward',total,flush=True)
    fig,axs=plt.subplots(3,4,figsize=(7.2,9.5),layout='constrained')
    for row,(img,ap,mp,p,q,after) in enumerate(views):
        panel(axs[row,0],img[:,:,:3].astype(np.uint8),f'Seed {records[row]["seed"]}: input')
        axs[row,0].scatter(p[1],p[0],s=30,c=AMBER,marker='x'); axs[row,0].scatter(q[1],q[0],s=25,facecolors='none',edgecolors='cyan')
        panel(axs[row,1],heat(ap),'Pick: log relative P','magma')
        panel(axs[row,2],heat(mp),'Place: max over angle','magma')
        panel(axs[row,3],after[:,:,:3].astype(np.uint8),f'After action: R={records[row]["reward"]:.0f}')
    save(fig,'model_rollouts')
    camera_rois=[]
    # Display crops from projected 16 cm neighborhoods of the two action points.
    first_img=views[0][0]; first_p=views[0][3]; first_q=views[0][4]
    centers=[np.asarray(utils.pix_to_xyz(v,first_img[:,:,3],agent.bounds,agent.pix_size)) for v in [first_p,first_q[:2]]]
    corners=np.stack([center+np.array([dx,dy,0]) for center in centers for dx in [-.08,.08] for dy in [-.08,.08]])
    for config,color in zip(agent.cam_config,obs_initial['color']):
        rot=np.asarray(pb.getMatrixFromQuaternion(config['rotation'])).reshape(3,3)
        camera_points=(corners-np.asarray(config['position']))@rot
        intrinsic=np.asarray(config['intrinsics']).reshape(3,3)
        projected=camera_points@intrinsic.T;uv=projected[:,:2]/projected[:,2:]
        lo=np.floor(uv.min(0)-8).astype(int);hi=np.ceil(uv.max(0)+8).astype(int)
        camera_rois.append([max(0,lo[1]),min(color.shape[0],hi[1]),max(0,lo[0]),min(color.shape[1],hi[0])])
    np.savez_compressed(results/'rollouts_arrays.npz',
        images=np.stack([v[0] for v in views]),pick_maps=np.stack([v[1] for v in views]),
        place_maps=np.stack([v[2] for v in views]),picks=np.asarray([v[3] for v in views]),
        places=np.asarray([v[4] for v in views]),after=np.stack([v[5] for v in views]),
        seeds=np.asarray([r['seed'] for r in records]),camera_rgb=np.stack(obs_initial['color']),camera_rois=np.asarray(camera_rois))
    img,ap,tp,p,q=initial
    fig,axs=plt.subplots(1,4,figsize=(8.2,2.6),layout='constrained')
    for i in range(3):panel(axs[i],obs_initial['color'][i],f'RGB-D camera {i+1}')
    panel(axs[3],img[:,:,:3].astype(np.uint8),'Fused top-down RGB')
    save(fig,'model_observation')

    # Pick-conditioned placing at three different interior red-object pixels.
    rgb=img[:,:,:3]; red=(rgb[:,:,0]>100)&(rgb[:,:,0]>rgb[:,:,1]*1.6)&(rgb[:,:,0]>rgb[:,:,2]*1.6)&(img[:,:,3]>0.02)
    candidates=np.argwhere(binary_erosion(red,iterations=2))
    assert len(candidates)>10
    picks=[np.asarray(p)]
    for _ in range(2):
        distances=np.min(np.stack([np.sum((candidates-v)**2,axis=1) for v in picks]),axis=0)
        picks.append(candidates[np.argmax(distances)])
    cond=[]; heatmaps=[]; curves=[]
    goal_T_object=utils.multiply(target_pose,utils.invert(object_pose))
    fig,axs=plt.subplots(2,3,figsize=(7.2,7.3),layout='constrained')
    for j,pj in enumerate(picks):
        _,tj,_,qj=maps(agent,img,pj)
        xyz=utils.pix_to_xyz(pj,img[:,:,3],agent.bounds,agent.pix_size)
        ideal=utils.multiply(goal_T_object,(xyz,utils.eulerXYZ_to_quatXYZW((0,0,0))))
        ideal_rc=utils.xyz_to_pix(ideal[0],agent.bounds,agent.pix_size)
        dist=float(np.linalg.norm(np.asarray(qj[:2])-ideal_rc)*agent.pix_size*1000)
        cond.append({'pick_rc':list(map(int,pj)),'predicted_place_rc_bin':list(qj),'ideal_place_rc':list(map(int,ideal_rc)),'position_error_mm':dist})
        heatmaps.append(tj.max(2)); curves.append(tj[qj[0],qj[1],:])
        panel(axs[0,j],rgb.astype(np.uint8),f'Anchor {j+1}')
        axs[0,j].scatter(pj[1],pj[0],c=AMBER,s=45,marker='x',label='pick')
        axs[0,j].scatter(qj[1],qj[0],edgecolors='cyan',facecolors='none',s=55,label='predicted place')
        axs[0,j].scatter(ideal_rc[1],ideal_rc[0],c='white',s=35,marker='+',label='rigid-body target')
        axs[0,j].add_patch(Rectangle((pj[1]-32,pj[0]-32),64,64,fill=False,ec=AMBER,lw=1))
        panel(axs[1,j],heat(tj.max(2)),f'Place map; error {dist:.1f} mm','magma')
    save(fig,'model_conditioned')
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.2),layout='constrained')
    for j,curve in enumerate(curves):
        axs[j].plot(np.arange(36)*10,curve/curve.max(),color=TEAL); axs[j].set_title(f'Anchor {j+1}, chosen pixel')
        axs[j].set_xlabel('Rotation bin (degrees)'); axs[j].set_ylim(-.03,1.05); axs[j].set_xticks([0,90,180,270]); axs[j].grid(alpha=.15)
    axs[0].set_ylabel('Relative score')
    save(fig,'model_angles')

    # Pixel shifts diagnose tensors only, without a new physical rollout.
    trans=[]; translation_maps=[]; translation_expected=[]; base=tp.max(2); shifts=[(0,0),(1,0),(0,1),(8,0),(0,8),(8,8)]
    diag=None
    for dr,dc in shifts:
        moved=shift(img,dr,dc); pnew=(p[0]+dr,p[1]+dc)
        am,tm,_,qm=maps(agent,moved,pnew)
        expect=shift(ap,dr,dc); valid=np.zeros(ap.shape,bool); valid[32:-32,32:-32]=True
        translation_maps.append(am); translation_expected.append(expect)
        a=am[valid]; b=expect[valid]; a=a/a.sum(); b=b/b.sum()
        tv=float(0.5*np.abs(a-b).sum())
        expected_q=np.asarray(q[:2])+[dr,dc]
        trans.append({'shift_rc':[dr,dc],'pick_total_variation_interior':tv,'place_peak_error_px':float(np.linalg.norm(np.asarray(qm[:2])-expected_q)),'place_bin':qm[2]})
        if (dr,dc)==(1,0):diag=(moved,am,expect)
    fig,axs=plt.subplots(1,4,figsize=(7.5,3.6),layout='constrained')
    panel(axs[0],heat(ap),'Original pick map','magma'); panel(axs[1],heat(diag[1]),'Input shifted by 1 row','magma')
    panel(axs[2],np.abs(diag[1]-diag[2]),'Absolute map difference','inferno')
    axs[3].bar(range(len(trans)),[x['pick_total_variation_interior'] for x in trans],color=TEAL)
    axs[3].set_xticks(range(len(trans)),[str(x['shift_rc']) for x in trans],rotation=70); axs[3].set_ylabel('Interior total variation'); axs[3].set_title('Finite network discrepancy')
    save(fig,'model_translation')

    # Local input-gradient sensitivity of key vs query. Crop is BEFORE query CNN.
    padded=np.pad(img,agent.transport.padding,mode='constant')
    normalized=agent.transport.preprocess(padded.copy())[None].astype(np.float32)
    x=tf.convert_to_tensor(normalized)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x); crop=x[:,p[0]:p[0]+64,p[1]:p[1]+64,:]
        key,query=agent.transport.model([x,crop],training=False)
        qs=tf.reduce_sum(tf.square(query[0,32,32,:]))
        ks=tf.reduce_sum(tf.square(key[0,q[0]+32,q[1]+32,:]))
    gq=np.linalg.norm(tape.gradient(qs,x).numpy()[0],axis=2)
    gk=np.linalg.norm(tape.gradient(ks,x).numpy()[0],axis=2); del tape
    inside=np.zeros(gq.shape,bool); inside[p[0]:p[0]+64,p[1]:p[1]+64]=True
    outside_fraction=float(gq[~inside].sum()/max(gq.sum(),1e-30))
    assert gq.sum()>0 and gk.sum()>0
    fig,axs=plt.subplots(1,3,figsize=(6.8,4),layout='constrained')
    panel(axs[0],rgb.astype(np.uint8),'Input and query crop')
    axs[0].add_patch(Rectangle((p[1]-32,p[0]-32),64,64,fill=False,ec=AMBER,lw=1.5))
    panel(axs[1],heat(gq[32:-32,32:-32]),'Query input sensitivity','magma')
    panel(axs[2],heat(gk[32:-32,32:-32]),'Key input sensitivity','magma')
    save(fig,'model_sensitivity')
    np.savez_compressed(results/'model_arrays.npz',image=img,pick=ap,place=tp,conditional_maps=np.stack(heatmaps),angle_curves=np.stack(curves),query_gradient=gq,key_gradient=gk,translation_maps=np.stack(translation_maps),translation_expected=np.stack(translation_expected))
    summary={'model':args.agent_name,'checkpoint':args.checkpoint,'devices':[d.device_type for d in tf.config.get_visible_devices()],'training_updates':0,'font_family':args.font_family,
             'rollouts':records,'conditioning':cond,'translation':trans,'query_gradient_fraction_outside_crop':outside_fraction,
             'elapsed_seconds':time.time()-started,'scope':'3 existing seeds re-run for visualization; conditional/translation/gradient diagnostics do not execute additional robot actions'}
    (results/'model_metrics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
finally:
    env.close()
