"""
Sistema de Inspeção Visual de Frutas — Interface Streamlit
Pré-requisito: executar notebooks 01-03 para gerar outputs/best_model.pkl
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from PIL import Image
import io
import datetime

# Importações de feature extraction (copiadas dos notebooks)
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, label as sk_label
from skimage.filters import threshold_otsu

# === REQUISITO BONUS: interface web em Streamlit ===
# ─── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Inspeção Visual de Frutas",
    page_icon="🍎",
    layout="wide",
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ─── Funções de segmentação (replicadas do notebook 01) ──────────────────────

def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """Segmenta a fruta via Otsu no canal S do HSV com fallback para cinza + anti-inversao."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Otsu no canal de saturação HSV ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    thresh_sat = threshold_otsu(sat)
    binary_sat = (sat > thresh_sat).astype(np.uint8) * 255
    sat_pct = np.count_nonzero(binary_sat) / binary_sat.size * 100

    if 5 < sat_pct < 70:
        # Saturação discrimina bem: fruta colorida vs fundo branco/cinza
        binary = binary_sat
    else:
        # Fallback: Otsu no cinza com anti-inversão por borda vs centro
        thresh_gray = threshold_otsu(blurred)
        binary = (blurred > thresh_gray).astype(np.uint8) * 255
        h, w = blurred.shape
        mg = max(5, min(h, w) // 8)
        border = np.concatenate([
            blurred[mg:h//4, mg:w-mg].flatten(),
            blurred[3*h//4:h-mg, mg:w-mg].flatten(),
            blurred[mg:h-mg, mg:w//4].flatten(),
            blurred[mg:h-mg, 3*w//4:w-mg].flatten()
        ])
        center = blurred[h//3:2*h//3, w//3:2*w//3].flatten()
        if len(border) > 0 and len(center) > 0 and border.mean() > center.mean():
            binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dilated

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(dilated)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica mascara binaria a imagem BGR, zerando pixels fora da regiao segmentada."""
    result = img_bgr.copy()
    result[mask == 0] = 0
    return result


# ─── Funções de extração de features (replicadas do notebook 02) ─────────────

def extract_shape_features(mask: np.ndarray) -> dict:
    """Extrai features geometricas da maior regiao da mascara (area, perimetro, excentricidade, solidez, extent, circularidade)."""
    binary = (mask > 0).astype(np.uint8)
    labeled = sk_label(binary)
    props = regionprops(labeled)
    if not props:
        return {k: 0.0 for k in ['area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'circularity']}
    region = max(props, key=lambda r: r.area)
    area = region.area
    perimeter = region.perimeter if region.perimeter > 0 else 1e-6
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'eccentricity': float(region.eccentricity),
        'solidity': float(region.solidity),
        'extent': float(region.extent),
        'circularity': float(np.clip((4 * np.pi * area) / (perimeter ** 2), 0, 1)),
    }


def extract_hu_moments(mask: np.ndarray) -> dict:
    """Calcula os 7 Momentos de Hu da mascara, transformados por log com sinal para estabilidade numerica."""
    binary = (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f'hu_{i+1}': float(v) for i, v in enumerate(hu_log)}


def extract_color_features(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Extrai estatisticas HSV (media/desvio por canal) e histograma de matiz (16 bins) dos pixels da fruta."""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    obj_pixels = img_hsv[mask > 0]
    if len(obj_pixels) == 0:
        feat = {c: 0.0 for c in ['h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std']}
        feat.update({f'h_hist_{i}': 0.0 for i in range(16)})
        return feat
    h, s, v = obj_pixels[:, 0], obj_pixels[:, 1], obj_pixels[:, 2]
    h_hist, _ = np.histogram(h, bins=16, range=(0, 180), density=True)
    h_hist = h_hist / (h_hist.sum() + 1e-10)
    feat = {
        'h_mean': float(np.mean(h)), 'h_std': float(np.std(h)),
        's_mean': float(np.mean(s)), 's_std': float(np.std(s)),
        'v_mean': float(np.mean(v)), 'v_std': float(np.std(v)),
    }
    feat.update({f'h_hist_{i}': float(v) for i, v in enumerate(h_hist)})
    return feat


def extract_glcm_features(img_gray: np.ndarray, mask: np.ndarray) -> dict:
    """Calcula features GLCM (contraste, homogeneidade, energia, correlacao) com distancia=1 em 4 angulos."""
    gray_masked = img_gray.copy()
    gray_masked[mask == 0] = 0
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_masked, distances=[1], angles=angles,
                        levels=256, symmetric=True, normed=True)
    return {
        'glcm_contrast':    float(graycoprops(glcm, 'contrast').mean()),
        'glcm_homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
        'glcm_energy':      float(graycoprops(glcm, 'energy').mean()),
        'glcm_correlation': float(graycoprops(glcm, 'correlation').mean()),
    }


def extract_lbp_features(img_gray: np.ndarray, mask: np.ndarray) -> dict:
    """Calcula histograma LBP uniform (P=24, R=3, 26 bins normalizados) sobre os pixels da mascara."""
    lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
    obj_lbp = lbp[mask > 0]
    hist, _ = np.histogram(obj_lbp, bins=26, range=(0, 26), density=True)
    hist = hist / (hist.sum() + 1e-10)
    return {f'lbp_{i}': float(v) for i, v in enumerate(hist)}


def extract_all_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Concatena todas as familias de features (forma, momentos Hu, cor HSV, GLCM, LBP) em um unico dicionario."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feat = {}
    feat.update(extract_shape_features(mask))
    feat.update(extract_hu_moments(mask))
    feat.update(extract_color_features(img_rgb, mask))
    feat.update(extract_glcm_features(img_gray, mask))
    feat.update(extract_lbp_features(img_gray, mask))
    return feat


# ─── Carregar modelo e metadados ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Carrega o modelo treinado (best_model.pkl), o scaler e o mapeamento de labels do diretorio outputs/."""
    model_path = OUTPUT_DIR / "best_model.pkl"
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    mapping_path = OUTPUT_DIR / "label_mapping.json"

    if not model_path.exists():
        return None, None, None, "Modelo não encontrado. Execute o notebook 03 primeiro."

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        with open(mapping_path) as f:
            label_mapping = json.load(f)
        return model, scaler, label_mapping, None
    except Exception as e:
        return None, None, None, f"Erro ao carregar modelo: {e}"


@st.cache_data
def load_results():
    """Carrega a tabela de resultados finais dos modelos (outputs/resultados_finais.csv) se disponivel."""
    results_path = OUTPUT_DIR / "resultados_finais.csv"
    if results_path.exists():
        return pd.read_csv(results_path, index_col=0)
    return None


# ─── UI ──────────────────────────────────────────────────────────────────────

st.title("Sistema de Inspeção Visual de Frutas")
st.markdown("**Inspeção automática de qualidade** — classifica frutas como frescas ou podres usando visão computacional.")

model, scaler, label_mapping, model_error = load_model()
df_results = load_results()

# Sidebar com informações do modelo
with st.sidebar:
    st.header("Informações do Modelo")

    if model_error:
        st.error(model_error)
    elif model is not None:
        model_name = type(model).__name__
        if hasattr(model, 'named_steps'):
            steps = list(model.named_steps.keys())
            model_name = type(model.named_steps[steps[-1]]).__name__
        st.success(f"Modelo carregado: **{model_name}**")

        if df_results is not None and model_name in df_results.index:
            f1 = df_results.loc[model_name, 'f1_macro']
            st.metric("F1-macro (test set)", f"{f1:.4f}")
        elif df_results is not None and len(df_results) > 0:
            best_row = df_results['f1_macro'].idxmax()
            st.metric(f"Melhor F1-macro ({best_row})", f"{df_results['f1_macro'].max():.4f}")

        st.info(f"Data de treino: 2026-05-25")

    st.divider()
    st.subheader("Classes")
    for cls in ['fresh_apple', 'rotten_apple', 'fresh_banana', 'rotten_banana',
                'fresh_orange', 'rotten_orange']:
        icon = "🟢" if "fresh" in cls else "🔴"
        st.markdown(f"{icon} {cls}")

# Upload de imagem
uploaded = st.file_uploader(
    "Envie uma imagem de fruta (JPG ou PNG)",
    type=["jpg", "jpeg", "png"],
    help="Imagem com fundo claro (branco ou preto) para melhor segmentação."
)

if uploaded is not None:
    if model_error:
        st.error(f"Não é possível realizar predição: {model_error}")
    else:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Imagem inválida. Envie um arquivo JPG ou PNG válido.")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)

            with st.spinner("Segmentando imagem..."):
                mask = segment_otsu(img_bgr)

                # Verificação de sanidade: máscara muito pequena ou quase total indica falha
                mask_pct = np.count_nonzero(mask) / mask.size * 100
                if mask_pct < 5:
                    st.warning(
                        f"⚠️ Segmentação provavelmente falhou: máscara cobre apenas {mask_pct:.1f}% da imagem. "
                        "Use uma foto com fundo claro (branco ou cinza) e a fruta centralizada."
                    )
                elif mask_pct > 95:
                    st.warning(
                        f"⚠️ Segmentação provavelmente falhou: máscara cobre {mask_pct:.1f}% da imagem (quase tudo). "
                        "Verifique se a imagem tem fundo bem distinto da fruta."
                    )

                segmented_bgr = apply_mask(img_bgr, mask)
                segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Segmentada (Otsu)")
                st.image(segmented_rgb, use_container_width=True)

            with st.spinner("Extraindo features..."):
                features = extract_all_features(img_bgr, mask)
                feature_vector = np.array(list(features.values())).reshape(1, -1)

                # Verificar se o scaler tem o número certo de features
                try:
                    # Se o modelo é uma Pipeline com scaler embutido, não aplicar scaler externo
                    if scaler is not None and not hasattr(model, 'steps'):
                        feature_vector_sc = scaler.transform(feature_vector)
                    else:
                        feature_vector_sc = feature_vector

                    proba = model.predict_proba(feature_vector_sc)[0]
                    pred_idx = np.argmax(proba)
                    pred_class = label_mapping.get(str(pred_idx), f"Classe {pred_idx}")
                    pred_prob = proba[pred_idx]
                    error_pred = None
                except Exception as e:
                    pred_class, pred_prob, error_pred = None, None, str(e)

            with col3:
                st.subheader("Predição")
                if error_pred:
                    st.error(f"Erro na predição: {error_pred}")
                else:
                    is_fresh = "fresh" in pred_class
                    status_icon = "✅" if is_fresh else "❌"
                    status_color = "green" if is_fresh else "red"

                    st.markdown(
                        f"<h2 style='color:{status_color};text-align:center'>"
                        f"{status_icon} {pred_class}</h2>",
                        unsafe_allow_html=True
                    )
                    st.progress(float(pred_prob))
                    st.caption(f"Confiança: {pred_prob * 100:.1f}%")

                    st.subheader("Top probabilidades")
                    top3_idx = np.argsort(proba)[::-1][:3]
                    for i in top3_idx:
                        cls_name = label_mapping.get(str(i), f"Classe {i}")
                        st.metric(label=cls_name, value=f"{proba[i] * 100:.1f}%")

            st.divider()
            st.subheader("Features Extraídas")
            df_feat = pd.DataFrame(list(features.items()), columns=["Feature", "Valor"])
            df_feat["Valor"] = df_feat["Valor"].apply(lambda x: round(x, 4))

            col_shape, col_color, col_texture = st.columns(3)
            shape_keys  = [k for k in features if any(k.startswith(p) for p in ['area','perimeter','eccentricity','solidity','extent','circularity','hu_'])]
            color_keys  = [k for k in features if any(k.startswith(p) for p in ['h_','s_','v_'])]
            texture_keys = [k for k in features if any(k.startswith(p) for p in ['glcm_','lbp_'])]

            with col_shape:
                st.markdown("**Forma & Momentos**")
                st.dataframe(df_feat[df_feat['Feature'].isin(shape_keys)].reset_index(drop=True),
                             height=300, hide_index=True)
            with col_color:
                st.markdown("**Cor HSV**")
                st.dataframe(df_feat[df_feat['Feature'].isin(color_keys)].reset_index(drop=True),
                             height=300, hide_index=True)
            with col_texture:
                st.markdown("**Textura (GLCM + LBP)**")
                st.dataframe(df_feat[df_feat['Feature'].isin(texture_keys)].reset_index(drop=True),
                             height=300, hide_index=True)

            # SHAP summary como referência
            shap_path = FIGURES_DIR / "shap_summary.png"
            if shap_path.exists():
                st.divider()
                st.subheader("Referência: Importância SHAP do Modelo")
                st.image(str(shap_path), caption="Features mais importantes segundo SHAP (notebook 03)")

elif model is None and model_error:
    st.warning(
        "Execute os notebooks 01, 02 e 03 para treinar o modelo e depois "
        "reinicie o app com: `streamlit run app/streamlit_app.py`"
    )
else:
    st.info("Envie uma imagem de fruta para começar a inspeção.")

    # Mostrar tabela de resultados se disponível
    if df_results is not None:
        st.divider()
        st.subheader("Resultados dos Modelos Treinados")
        st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
