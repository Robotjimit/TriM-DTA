"""Reproducible computational-cost and KIBA-scale experiment for TriM-DTA.

This entry point deliberately reuses the repository's DMFF model and data
representation.  It only adds fixed nested sampling plus synchronized timing
and memory accounting; it does not alter model, loss, or optimization logic.
"""
import argparse, json, os, platform, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index

# Load the original imports and DMFF definition, stopping before main.py's
# unguarded data preparation/training program.
_src = Path('main.py').read_text()
_model_src = _src[:_src.index("df = pd.read_csv")]
_ns = {'__name__': '_trim_model_definition_'}
exec(compile(_model_src, 'main.py', 'exec'), _ns)
DMFF = _ns['DMFF']
smiles_to_graph = _ns['smiles_to_graph']
from build_vocab import WordVocab
from dataset import DTA_Dataset
from utils import get_mse


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_features(dataset):
    df = pd.read_csv(f'{dataset}/{dataset}_processed.csv')
    smiles, targets = set(df.compound_iso_smiles), set(df.target_key)
    target_seq = dict(zip(df.target_key, df.target_sequence))
    smiles_graph = {s: smiles_to_graph(s)[1] for s in smiles}
    target_graph = {}
    contact_dir = Path(f'target_contact_map_{dataset}')
    for target, seq in target_seq.items():
        row = df[df.target_key == target].iloc[0]
        name = target if dataset == 'kiba' else row.uniprot
        cm = np.load(contact_dir / f'{name}.npy')
        cm = cm[int(row.target_sequence_start):int(row.target_sequence_end), int(row.target_sequence_start):int(row.target_sequence_end)]
        target_graph[target] = np.column_stack(np.where(cm > .8))
    drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
    protein_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')
    smiles_emb, smiles_len, smiles_idx = {}, {}, {}
    for sm in smiles:
        vals=[]; i=0
        while i < len(sm):
            if i+1 < len(sm) and sm[i:i+2] in drug_vocab.stoi: vals.append(drug_vocab.stoi[sm[i:i+2]]); i += 2
            else: vals.append(drug_vocab.stoi.get(sm[i], drug_vocab.unk_index)); i += 1
        vals=vals[:540]; smiles_len[sm]=len(vals); x=[drug_vocab.sos_index]+vals+[drug_vocab.eos_index]
        x += [drug_vocab.pad_index] * (540-len(x)); smiles_emb[sm]=torch.tensor(x)
        smiles_idx[sm] = [j for j, v in enumerate(x) if v in _ns['atom_dict']]
    target_emb, target_len = {}, {}
    for key, seq in target_seq.items():
        vals=[]; i=0
        while i < len(seq):
            if i+1 < len(seq) and seq[i:i+2] in protein_vocab.stoi: vals.append(protein_vocab.stoi[seq[i:i+2]]); i += 2
            else: vals.append(protein_vocab.stoi.get(seq[i], protein_vocab.unk_index)); i += 1
        vals=vals[:1000]; target_len[seq]=len(vals); x=[protein_vocab.sos_index]+vals+[protein_vocab.eos_index]
        x += [protein_vocab.pad_index] * (1000-len(x)); target_emb[seq]=torch.tensor(x)
    return smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len


def build_dataset(path, features, dataset, mode='default', row_indices=None):
    graph_dir = f'./{dataset}/pyg' if dataset == 'kiba' else f'./{dataset}/pyg_8'
    return DTA_Dataset('./', path, *features, mode=mode, graph_dir=graph_dir, row_indices=row_indices)


def move(batch, device): return [x.to(device) for x in batch]


def train_epoch(model, loader, optimizer, device):
    model.train(); loss_fn=torch.nn.MSELoss()
    for batch in loader:
        batch=move(batch, device); optimizer.zero_grad(set_to_none=True)
        out, y=model(batch); loss=loss_fn(out.float(), y.float().to(device)); loss.backward(); optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for batch in loader:
        out,y=model(move(batch,device)); ys.append(y.view(-1).cpu()); ps.append(out.view(-1).cpu())
    y=torch.cat(ys).numpy(); p=torch.cat(ps).numpy()
    # This is the same CI implementation explicitly used by main.py for its
    # periodic KIBA test reporting, unlike utils.get_ci's O(n^2) Python loop.
    return float(get_mse(y,p)), float(concordance_index(y,p))


def timed_epoch(model, loader, optimizer, device):
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); t=time.perf_counter()
    train_epoch(model, loader, optimizer, device)
    torch.cuda.synchronize(device)
    return time.perf_counter()-t, torch.cuda.max_memory_allocated(device)/1024**2


@torch.no_grad()
def timed_inference(model, dataset, batch_size, device, repeats, warmup_batches):
    # Batches are collated before the clock; each repetition is a full test pass.
    warm = list(DataLoader(dataset, batch_size=batch_size, shuffle=False))[:warmup_batches]
    model.eval()
    for b in warm: model(move(b,device))
    torch.cuda.synchronize(device)
    seconds=[]
    for _ in range(repeats):
        batches=list(DataLoader(dataset,batch_size=batch_size,shuffle=False))
        total=0.0
        for b in batches:
            b=move(b,device); torch.cuda.synchronize(device); t=time.perf_counter(); model(b); torch.cuda.synchronize(device); total += time.perf_counter()-t
        seconds.append(total / len(dataset) * 1000)
    return seconds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='kiba', choices=['kiba','davis'])
    ap.add_argument('--mode', default='default'); ap.add_argument('--dim',type=int,default=128)
    ap.add_argument('--batch-size',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--timing-epochs',type=int,default=3)
    ap.add_argument('--fraction',type=float,default=1.0); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--inference-repeats',type=int,default=3); ap.add_argument('--out',default='experiment_results.json')
    ap.add_argument('--best-checkpoint', default=None,
                    help='Path to overwrite whenever validation MSE improves.')
    a=ap.parse_args(); assert torch.cuda.is_available(), 'CUDA is required for this experiment'
    seed_everything(a.seed); device=torch.device('cuda:0')
    print('stage=features', flush=True); features=make_features(a.dataset)
    train_path=f'./{a.dataset}/{a.mode}/train.csv'
    train_count=len(pd.read_csv(train_path))
    order=np.random.RandomState(a.seed).permutation(train_count); n=int(train_count*a.fraction)
    print('stage=train_dataset', flush=True); train_full=build_dataset(train_path,features,a.dataset,a.mode,order[:n].tolist())
    print('stage=valid_dataset', flush=True); valid=build_dataset(f'./{a.dataset}/{a.mode}/valid.csv',features,a.dataset,a.mode)
    print('stage=test_dataset', flush=True); test=build_dataset(f'./{a.dataset}/{a.mode}/test.csv',features,a.dataset,a.mode)
    train_loader=DataLoader(train_full,batch_size=a.batch_size,shuffle=True)
    valid_loader=DataLoader(valid,batch_size=a.batch_size,shuffle=False); test_loader=DataLoader(test,batch_size=a.batch_size,shuffle=False)
    print('stage=model', flush=True); model=DMFF(a.dim*2,a.dim,a.dim*2,.2,.2,8,2,26,45,.5).to(device)
    # Intentionally start from the deterministic random initialization above.
    # Loading a previous KIBA model would invalidate the requested comparison
    # of predictive performance across training-set sizes.
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    # Warm-up epoch is excluded from reported runtime and peak-memory measurements.
    print('stage=warmup', flush=True); train_epoch(model,train_loader,optimizer,device); torch.cuda.synchronize(device)
    epoch_times=[]; peaks=[]; best=(float('inf'),None)
    for epoch in range(a.epochs):
        if epoch < a.timing_epochs:
            elapsed,peak=timed_epoch(model,train_loader,optimizer,device); epoch_times.append(elapsed); peaks.append(peak)
        else: train_epoch(model,train_loader,optimizer,device)
        mse,_=evaluate(model,valid_loader,device)
        if mse < best[0]:
            best=(mse,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if a.best_checkpoint:
                checkpoint_path=Path(a.best_checkpoint)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch + 1, 'valid_mse': mse,
                            'state_dict': best[1], 'seed': a.seed}, checkpoint_path)
        print(f'stage=epoch epoch={epoch + 1}/{a.epochs} valid_mse={mse:.6f}', flush=True)
    if best[1] is not None: model.load_state_dict(best[1])
    mse,ci=evaluate(model,test_loader,device)
    inf=timed_inference(model,test,a.batch_size,device,a.inference_repeats,2)
    prop=torch.cuda.get_device_properties(device)
    result={'dataset':a.dataset,'mode':a.mode,'fraction':a.fraction,'seed':a.seed,'train_interactions':n,
      'valid_interactions':len(valid),'test_interactions':len(test),'batch_size':a.batch_size,'epochs':a.epochs,
      'timing_epochs':a.timing_epochs,'params':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'epoch_seconds':epoch_times,'epoch_mean_s':float(np.mean(epoch_times)),'epoch_std_s':float(np.std(epoch_times,ddof=1)) if len(epoch_times)>1 else 0.,
      'peak_memory_mb':float(max(peaks)),'peak_memory_by_epoch_mb':peaks,'inference_ms_per_sample':inf,
      'inference_mean_ms':float(np.mean(inf)),'inference_std_ms':float(np.std(inf,ddof=1)) if len(inf)>1 else 0.,'test_mse':mse,'test_ci':ci,
      'best_checkpoint':a.best_checkpoint,
      'gpu':prop.name,'gpu_total_memory_mb':prop.total_memory/1024**2,'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version()}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__ == '__main__': main()
