"""Os três candidatos, todos com o mesmo pré-processamento (ColumnTransformer dentro do Pipeline = sem leakage)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.base_analitica import CAT, NUM


def _preparo(escala: bool) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median", add_indicator=True)]   # nulo vira flag (ex.: nunca teve CPC)
    if escala:   # modelos lineares e redes precisam de escala; árvores não
        num = [FunctionTransformer(np.log1p, feature_names_out="one-to-one")] + num + [StandardScaler()]
    ct = ColumnTransformer([("num", make_pipeline(*num), NUM),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return ct.set_output(transform="pandas")   # nomes de coluna chegam ao modelo (importância legível)


def criar_modelos() -> dict[str, Pipeline]:
    return {
        "logistica": make_pipeline(_preparo(True), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "lightgbm": make_pipeline(_preparo(False), LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)),
        "mlp": make_pipeline(_preparo(True), MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42, max_iter=300)),
    }
