import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ============================================================
# AI OUTFIT COMPATIBILITY
# Experiment 6 — ResNet50 + Two-Tower Network
# ============================================================

st.set_page_config(
    page_title="AI Outfit Compatibility",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESNET_WEIGHTS = BASE_DIR / "resnet50-11ad3fa6.pth"
MODEL_PATH = BASE_DIR / "resnet50_two_tower_epoch6.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# RESNET50 FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_resnet():

    if not RESNET_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Missing {RESNET_WEIGHTS.name}"
        )

    # Load the original ImageNet ResNet50 first
    model = resnet50(weights=None)

    state_dict = torch.load(
        RESNET_WEIGHTS,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    # Remove ImageNet classification layer
    # and use ResNet50 as a feature extractor.
    model.fc = nn.Identity()

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# TWO-TOWER MODEL
# ============================================================

class ResNet50TwoTower(nn.Module):

    def __init__(
        self,
        input_dim=2048,
        embedding_dim=128
    ):

        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                512,
                embedding_dim
            )
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim * 4,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        top,
        bottom
    ):

        top_proj = self.projection(top)

        bottom_proj = self.projection(bottom)

        difference = torch.abs(
            top_proj - bottom_proj
        )

        product = top_proj * bottom_proj

        combined = torch.cat(
            [
                top_proj,
                bottom_proj,
                difference,
                product
            ],
            dim=1
        )

        return self.classifier(combined)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_compatibility_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ResNet50TwoTower(
        input_dim=checkpoint.get(
            "input_dim",
            2048
        ),
        embedding_dim=checkpoint.get(
            "embedding_dim",
            128
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_compatibility(
    top_image,
    bottom_image
):

    feature_extractor = load_resnet()

    compatibility_model = (
        load_compatibility_model()
    )

    top_tensor = IMAGE_TRANSFORM(
        top_image
    ).unsqueeze(0).to(DEVICE)

    bottom_tensor = IMAGE_TRANSFORM(
        bottom_image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        top_features = feature_extractor(
            top_tensor
        )

        bottom_features = feature_extractor(
            bottom_tensor
        )

        logit = logits.item()

        probability = torch.sigmoid(logits).item()

    prediction = (
        "Compatible"
        if probability >= 0.5
        else "Incompatible"
    )

    return probability, prediction,logit


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #777777;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .score {
        font-size: 3rem;
        font-weight: 800;
    }

    .model-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f7f7f7;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '👗 AI Outfit Compatibility'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Fashion Compatibility Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 About the Model")

    st.write(
        """
        This application uses the trained
        **Experiment 6 ResNet50 + Two-Tower
        compatibility network**.
        """
    )

    st.markdown("### Architecture")

    st.write(
        """
        **ResNet50**

        ↓

        **2048-D visual features**

        ↓

        **Shared 128-D embeddings**

        ↓

        **Pairwise relationship features**

        ↓

        **Neural classifier**
        """
    )

    st.markdown("### Test Benchmark")

    st.metric(
        "Accuracy",
        "96.52%"
    )

    st.metric(
        "F1 Score",
        "96.47%"
    )

    st.metric(
        "ROC-AUC",
        "0.9930"
    )

    st.caption(
        "Results are from the held-out benchmark "
        "with constructed hard negatives."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👕 Choose Your Outfit'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload one top and one bottom to evaluate the outfit."
)

top_col, bottom_col = st.columns(
    2,
    gap="large"
)


with top_col:

    st.subheader("👕 TOP")

    top_file = st.file_uploader(
        "Upload a top image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="top"
    )

    if top_file:

        top_image = Image.open(
            top_file
        ).convert("RGB")

        st.image(
            top_image,
            caption="Selected Top",
            use_container_width=True
        )


with bottom_col:

    st.subheader("👖 BOTTOM")

    bottom_file = st.file_uploader(
        "Upload a bottom image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="bottom"
    )

    if bottom_file:

        bottom_image = Image.open(
            bottom_file
        ).convert("RGB")

        st.image(
            bottom_image,
            caption="Selected Bottom",
            use_container_width=True
        )


st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

check_button = st.button(
    "🔍  CHECK OUTFIT COMPATIBILITY",
    type="primary",
    use_container_width=True
)


if check_button:

    if (
        top_file is None
        or bottom_file is None
    ):

        st.warning(
            "Please upload both a top and a bottom image."
        )

    else:

        with st.spinner(
            "Analyzing outfit compatibility..."
        ):

            try:

                probability, prediction,logit = (
                    predict_compatibility(
                        top_image,
                        bottom_image
                    )
                )

                score = probability * 100
                st.write(f"Raw model logit: {logit:.6f}")
                st.write(f"Raw sigmoid probability: {probability:.10f}")

                st.divider()

                st.markdown(
                    "## 🎯 Prediction Result"
                )

                result_col1, result_col2 = (
                    st.columns(2)
                )

                with result_col1:

                    st.metric(
                        "Compatibility Score",
                        f"{score:.2f}%"
                    )

                with result_col2:

                    if prediction == "Compatible":

                        st.success(
                            "🟢 COMPATIBLE"
                        )

                    else:

                        st.error(
                            "🔴 INCOMPATIBLE"
                        )

                st.progress(
                    probability
                )

                st.caption(
                    "The score represents the model's "
                    "predicted compatibility probability."
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.divider()

with st.expander(
    "🔬 Technical Details"
):

    st.markdown(
        """
        ### Model Pipeline

        **Input images**

        → Resize to 224 × 224

        → ImageNet normalization

        → ResNet50 feature extraction

        → 2048-dimensional visual features

        → Shared projection network

        → 128-dimensional embeddings

        → Pairwise relationship features

        → Neural classifier

        → Compatibility probability

        ### Pairwise Features

        The model uses:

        - Top embedding
        - Bottom embedding
        - Absolute difference
        - Element-wise product
        """
    )


with st.expander(
    "📊 Experiment 6 Results"
):

    st.markdown(
        """
        **Held-out test benchmark**

        | Metric | Result |
        |---|---:|
        | Accuracy | 96.52% |
        | Precision | 97.98% |
        | Recall | 95.00% |
        | F1-score | 96.47% |
        | ROC-AUC | 0.9930 |

        **Important:** The benchmark uses constructed hard negatives,
        rather than independently human-annotated incompatibility labels.
        Therefore these metrics describe performance on the project's
        held-out benchmark.
        """
    )


st.caption(
    "Fashion Compatibility Project • "
    "Experiment 6 • ResNet50 Two-Tower"
)
