import io
from typing import Any

import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Cow Classifier",
    page_icon="🐄",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            }
            .block-container {
                max-width: 1100px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .hero {
                background: white;
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
                border: 1px solid #e2e8f0;
                margin-bottom: 1rem;
            }
            .result-card {
                background: white;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            }
            .metric-label {
                font-size: 0.85rem;
                color: #475569;
            }
            .metric-value {
                font-size: 1.2rem;
                font-weight: 700;
                color: #0f172a;
            }
            .small-note {
                color: #64748b;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(base_url: str, endpoint: str, timeout: int = 30) -> tuple[bool, Any]:
    try:
        response = requests.get(f"{base_url}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def api_identify(base_url: str, image_bytes: bytes, filename: str, threshold: float, timeout: int = 60) -> tuple[bool, Any]:
    files = {"image": (filename, image_bytes, "image/jpeg")}
    try:
        response = requests.post(
            f"{base_url}/cows/classify",
            files=files,
            params={"confidence_threshold": threshold},
            timeout=timeout,
        )
        response.raise_for_status()
        return True, response.json()
    except requests.HTTPError:
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
            return False, f"Erro API ({response.status_code}): {detail}"
        except Exception:  # noqa: BLE001
            return False, f"Erro API ({response.status_code}): {response.text}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def api_get_reference_image(base_url: str, cow_id: int, timeout: int = 30) -> tuple[bool, Any]:
    try:
        response = requests.get(f"{base_url}/cows/{cow_id}/image", timeout=timeout)
        response.raise_for_status()
        return True, response.content
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def sidebar_settings() -> tuple[str, float, int]:
    st.sidebar.title("⚙️ Configurações")
    base_url = st.sidebar.text_input("URL da API", value="http://localhost:8000").rstrip("/")
    threshold = st.sidebar.slider("Limiar de similaridade", min_value=0.50, max_value=0.999, value=0.98, step=0.005)
    timeout = st.sidebar.slider("Timeout (segundos)", min_value=10, max_value=180, value=60, step=5)

    st.sidebar.markdown("---")
    st.sidebar.caption("Fluxo: envia foto → endpoint /cows/classify → retorna classe conhecida ou desconhecida.")

    if st.sidebar.button("Testar conexão"):
        ok, payload = api_get(base_url, "/cows", timeout=min(timeout, 30))
        if ok:
            items = payload.get("items", []) if isinstance(payload, dict) else []
            st.sidebar.success(f"API online. Registros na base: {len(items)}")
        else:
            st.sidebar.error(f"Falha na conexão: {payload}")

    return base_url, threshold, timeout


def friendly_reason(reason: str | None) -> tuple[str, str]:
    reason_map: dict[str, tuple[str, str]] = {
        "recognized": ("success", "Vaca identificada com sucesso na base cadastrada."),
        "no_keypoints_detected": (
            "info",
            "Não foi possível detectar os pontos-chave da vaca nesta imagem. Tente outra foto com melhor enquadramento e iluminação.",
        ),
        "partial_keypoints_detected": (
            "info",
            "A imagem gerou apenas parte dos pontos-chave necessários. Tente uma foto com a vaca mais visível e sem obstruções.",
        ),
        "empty_database": (
            "warning",
            "A base está vazia. Cadastre vacas antes de realizar a identificação.",
        ),
        "below_similarity_threshold": (
            "warning",
            "A vaca não atingiu o limiar de similaridade e foi classificada como desconhecida.",
        ),
        "below_confidence_threshold": (
            "warning",
            "A confiança da classificação ficou abaixo do limiar e a vaca foi classificada como desconhecida.",
        ),
    }
    return reason_map.get(
        reason or "",
        ("warning", "A vaca foi classificada como desconhecida para os parâmetros atuais."),
    )


def main() -> None:
    inject_styles()
    base_url, threshold, timeout = sidebar_settings()

    st.markdown(
        """
        <div class='hero'>
            <h2 style='margin:0;'>🐄 Identificação de Vacas</h2>
            <p style='margin:0.35rem 0 0 0;' class='small-note'>
                Envie uma foto de vaca e receba o resultado da identificação pela API:
                classe conhecida na base ou <b>desconhecida</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    with col_left:
        uploaded = st.file_uploader(
            "Selecione uma imagem da vaca",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Pré-visualização", use_container_width=True)

            if st.button("🔎 Identificar vaca", type="primary", use_container_width=True):
                with st.spinner("Classificando imagem..."):
                    image_bytes = uploaded.getvalue()
                    ok, payload = api_identify(
                        base_url=base_url,
                        image_bytes=image_bytes,
                        filename=uploaded.name,
                        threshold=threshold,
                        timeout=timeout,
                    )

                if not ok:
                    st.error(f"Não foi possível classificar a imagem. {payload}")
                    return

                st.session_state["last_result"] = payload
                st.session_state["last_base_url"] = base_url

    with col_right:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Resultado")

        result = st.session_state.get("last_result")
        result_base_url = st.session_state.get("last_base_url", base_url)

        if not result:
            st.info("Envie uma foto e clique em “Identificar vaca” para ver o resultado.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        recognized = bool(result.get("recognized", False))
        similarity = result.get("confidence")
        matched_id = result.get("predicted_class")
        used_threshold = result.get("threshold")
        reason = result.get("reason")

        m1, m2 = st.columns(2)
        with m1:
            status = "Identificada" if recognized else "Desconhecida"
            st.markdown(f"<div class='metric-label'>Status</div><div class='metric-value'>{status}</div>", unsafe_allow_html=True)
        with m2:
            sim_text = "-" if similarity is None else f"{float(similarity):.4f}"
            st.markdown(f"<div class='metric-label'>Similaridade</div><div class='metric-value'>{sim_text}</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        if recognized and matched_id is not None:
            st.success(f"Vaca reconhecida: classe **{matched_id}**")
            st.caption(f"Limiar aplicado: {used_threshold}")

            if isinstance(matched_id, int) or (isinstance(matched_id, str) and matched_id.isdigit()):
                ok_ref, ref_payload = api_get_reference_image(result_base_url, int(matched_id), timeout=min(timeout, 30))
                if ok_ref:
                    st.image(Image.open(io.BytesIO(ref_payload)), caption=f"Imagem de referência da vaca {matched_id}", use_container_width=True)
        else:
            level, message = friendly_reason(reason)
            if level == "info":
                st.info(message)
            elif level == "success":
                st.success(message)
            else:
                st.warning(message)
            if reason:
                st.caption(f"Motivo técnico: `{reason}`")
            st.caption(f"Limiar aplicado: {used_threshold}")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
