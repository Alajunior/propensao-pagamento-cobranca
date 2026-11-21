import numpy as np

from src.calibracao import aplicar_calibracao, ajustar_calibracao, brier, ece, previu_pagou


def test_calibracao_preserva_ordem_e_acerta_media():
    rng = np.random.default_rng(0)
    n = 20000
    fase = rng.choice(["0001-0360", "0361-0720"], n)
    p_real = rng.uniform(0.01, 0.2, n)
    y = (rng.random(n) < p_real).astype(int)
    score = np.clip(p_real * 4, 0, 1)                    # ordena certo, número inflado (como o balanced)
    cal = ajustar_calibracao(score, fase, y)
    p = aplicar_calibracao(cal, score, fase)
    assert abs(p.mean() - y.mean()) < 0.005
    assert ece(y, p) < ece(y, score) and brier(y, p) < brier(y, score)
    for f in np.unique(fase):
        m = fase == f
        assert np.all(np.diff(p[m][np.argsort(score[m])]) >= -1e-12)   # monotônica dentro da fase


def test_fase_pequena_usa_global():
    score = np.linspace(0, 1, 1000)
    y = (score > 0.9).astype(int)
    fase = np.array(["0001-0360"] * 990 + ["1081+"] * 10)
    cal = ajustar_calibracao(score, fase, y)
    assert "1081+" not in cal and "__GLOBAL__" in cal
    assert np.isfinite(aplicar_calibracao(cal, score, fase)).all()


def test_previu_pagou_tem_10_decis():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    t = previu_pagou(y, p)
    assert list(t.decil) == list(range(1, 11)) and t.n.sum() == 5000
    assert t.previsto.iloc[0] > t.previsto.iloc[-1]
