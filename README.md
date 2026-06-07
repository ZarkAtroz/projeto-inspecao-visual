# Inspeção Visual Automática de Frutas

## Descrição

Sistema de visão computacional para classificação automática de qualidade de frutas (fresca vs podre). O pipeline combina segmentação por Otsu, extração de features clássicas (cor HSV, textura GLCM/LBP, forma) e classificadores treinados com GridSearchCV. Inclui bônus com MobileNetV2 + Grad-CAM e interface interativa via Streamlit.

## Dataset

- **Nome:** Fruit Quality Detection (Kaggle)
- **Link:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- **Classes:** fresh_apple, rotten_apple, fresh_banana, rotten_banana, fresh_orange, rotten_orange
- **Imagens por classe:** até 200 (balanceado por undersampling)

## Pipeline

```
data/fruits/<classe>/
        │
        ▼
[01] Segmentação (Otsu)
        │
        ▼
data/segmented/<classe>/
        │
        ▼
[02] Extração de Features
     Shape + Hu Moments + HSV + GLCM + LBP
        │
        ▼
outputs/X.csv  +  outputs/y.csv
        │
        ▼
[03] Classificação (SVM / RF / LR)
     GridSearchCV + SHAP + Ablation Study
        │
        ▼
[04] CNN Bônus: MobileNetV2 + Grad-CAM
        │
        ▼
[app] Streamlit — Interface de Demo
```

## Instalação e Execução

```bash
pip install -r requirements.txt

# Baixar dataset via Kaggle API
kaggle datasets download sriramr/fruits-fresh-and-rotten-for-classification
unzip fruits-fresh-and-rotten-for-classification.zip -d data/fruits/

# Executar notebooks em ordem:
jupyter notebook notebooks/01_segmentacao.ipynb
jupyter notebook notebooks/02_features.ipynb
jupyter notebook notebooks/03_classificacao.ipynb
jupyter notebook notebooks/04_cnn_xai_bonus.ipynb  # bônus

# Iniciar interface web:
streamlit run app/streamlit_app.py
```

## Resultados

### Modelos Clássicos vs CNN — Test Set (70/15/15, stratificado)

| Modelo | Accuracy | F1-macro | AUC |
|--------|----------|----------|-----|
| **Random Forest** *(melhor clássico)* | **95.0%** | **0.950** | **0.995** |
| SVM (RBF, C=10) | 93.9% | 0.939 | 0.993 |
| Logistic Regression (L2) | 91.1% | 0.911 | 0.988 |
| **MobileNetV2** *(bônus)* | **97.2%** | **0.972** | * |

> \* AUC não calculado para MobileNetV2 — métrica aplicada apenas aos modelos clássicos com predict_proba via one-vs-rest.

### Ablation Study — F1-macro por Grupo de Features (CV=5, X_train)

| Grupo de Features | LR | RF | SVM |
|---|---|---|---|
| G1: Cor (HSV) | 0.643 | 0.831 | 0.742 |
| G2: Textura (GLCM+LBP) | 0.819 | 0.835 | 0.828 |
| G3: Forma (Shape+Hu) | 0.532 | 0.660 | 0.556 |
| **G4: Todos** | **0.906** | **0.912** | **0.900** |

> **Conclusão:** features de cor e textura sao individualmente as mais discriminativas; a combinacao de todas as familias alcanca o melhor resultado nos modelos classicos. A CNN MobileNetV2 superou os classicos em +2.2% de F1-macro (0.972 vs 0.950) aprendendo representacoes diretamente dos pixels.

## Estrutura do Repositório

```
projeto-inspecao-visual/
├── data/
│   ├── fruits/                 # Dataset original por classe
│   └── segmented/              # Imagens após segmentação
├── notebooks/
│   ├── 01_segmentacao.ipynb    # EDA + Otsu + Watershed
│   ├── 02_features.ipynb       # Shape + HSV + GLCM + LBP + PCA
│   ├── 03_classificacao.ipynb  # SVM/RF/LR + SHAP + Ablation
│   └── 04_cnn_xai_bonus.ipynb  # MobileNetV2 + Grad-CAM
├── outputs/
│   ├── figures/                # Todas as figuras geradas
│   ├── X.csv                   # Feature matrix
│   ├── y.csv                   # Labels
│   ├── best_model.pkl          # Melhor modelo salvo
│   └── resultados_finais.csv   # Tabela comparativa
├── app/
│   └── streamlit_app.py        # Interface web
├── README.md
└── requirements.txt
```

## Notas Metodologicas

O dataset Kaggle continha variantes augmentadas (rotacoes e ruido sal-e-pimenta).
Removemos essas variantes para evitar data leakage; o pipeline utiliza apenas
imagens originais. Os resultados reportados refletem essa configuracao limpa.

## Equipe

| Integrante | Responsabilidade |
|---|---|
| Felipe de Mello Vieira | Notebooks 01 + 02 — segmentação, extração de features, EDA |
| Felipe Yukiya Soares Uemura | Apoio na revisao e documentacao |
| Gabriel Sampaio Giacomoni | Notebook 03 — classificação, SHAP, ablation study, relatório PDF |
| Leonardo Gabriel Herédia | Notebook 04 + Streamlit + vídeo demo + README |
