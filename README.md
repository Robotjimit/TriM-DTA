# title

TriM-DTA: A tri-modal fusion framework for drug–target binding affinity prediction

## overview

![overview](fig/legend.png)

## abstract

Predicting drug–target binding affinity is a crucial problem in computational drug discovery, requiring accurate modeling of diverse molecular representations. Existing approaches often rely on a single modality, such as sequence, graph, or structure, they have limitations in capturing the complementary biochemical and spatial cues behind molecular interactions. In this work, we propose TriM-DTA, a tri-modal information fusion framework to accurate predict drug-target binding affinity that integrates sequence features, topological graphs, and geometric structures of both drugs and targets. This framework consists of modality-specific encoders and a cross-modal attention fusion module that jointly learns affinity-aware representations by aligning structural and sequence information. We evaluate TriM-DTA on two benchmark datasets under both seen and unseen scenarios, where it consistently outperforms state-of-the
art methods in predictive accuracy and generalization. Ablation studies confirm the distinct contribution of each modality to overall performance. Furthermore, embedding space analysis reveals that the model organizes molecular representations into well-separated clusters aligned with binding strength. Atomic level visualization highlights chemically meaningful substructures at protein–ligand interfaces, supporting the interpretability and biological plausibility of TriM-DTA. These results demonstrate that tri-modal fusion provides a unified and expressive view of drug-target binding affinity prediction. TriM-DTA offers a flexible and extensible foundation for structure-aware molecular modeling and holds promise in binding pose prediction, selectivity estimation, and mechanism-driven drug design. 

### install

```bash
pip install -r requirements.txt
```

### dataset

coming soon

### quickly start


```bash
python main.py
```

## file

```
.
├── config.py
├── egnn.py
├── main.py
├── model.py
├── utils.py
└── __pycache__/
```

## cite
```
coming soon
```

